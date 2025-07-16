/*
  Simple examples for CKKS
 */

#define PROFILE

#include "openfhe.h"
#include <random>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <thread>

#include "globals.hpp"

extern "C" {
    #include "nn.h"
    #include "train.h"
}

// Sigmoid: 1 / (1 + exp(-x))
auto sigmoid_chebyshev = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
};

using namespace lbcrypto;

Ciphertext<DCRTPoly> forward_pass(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    const std::vector<Ciphertext<DCRTPoly>>& cx,
    int input_size,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cA,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cZ
    ){

    std::chrono::high_resolution_clock::time_point denb;

    int hardware_threads = std::thread::hardware_concurrency();
    int omp_threads = omp_get_max_threads();

    int extra_threads = hardware_threads - omp_threads;
    int available_threads = std::max(1, extra_threads);

    denb = std::chrono::high_resolution_clock::now();
    for(int i=0; i<input_size; i++){
        cA[0][i] = cx[i];
    }

    Ciphertext<DCRTPoly> cY;

    std::vector<double> bat = {1.0};
    Plaintext pBat = cc->MakeCKKSPackedPlaintext(bat, 1);

    for(int layer = 1; layer < nn->n_layers; layer++){
        //std::cout << "\t\tLayer: " << layer << std::endl;
        //#pragma omp parallel for num_threads(available_threads)
        for(int neurona = 0; neurona < nn->layers_size[layer]; neurona++){
            //std::cout << "\t\tNeurona: " << neurona << std::endl;
            // Zlj
            #pragma omp parallel for num_threads(available_threads)
            for(size_t k=0; k<cW[layer-1][neurona].size(); k++){
                if(depth - cW[layer-1][neurona][k]->GetLevel() < 6)
                    cW[layer-1][neurona][k] = cc->EvalBootstrap(cW[layer-1][neurona][k]);
            }
            // wx + b
            cZ[layer][neurona] = cB[layer-1][neurona];
            for(int j=0; j<nn->layers_size[layer-1]; j++){
                Ciphertext<DCRTPoly> cMult = cc->EvalMult(cA[layer-1][j], cW[layer-1][neurona][j]);

                if(depth - cMult->GetLevel() < 6)
                    cMult = cc->EvalBootstrap(cMult);

                cZ[layer][neurona] = cc->EvalAdd(cZ[layer][neurona], cMult);
            }

            if(depth - cZ[layer][neurona]->GetLevel() < 8){
                cZ[layer][neurona] = cc->EvalBootstrap(cZ[layer][neurona]);
            }

            // Sigmoid
            cA[layer][neurona] = cc->EvalChebyshevFunction(sigmoid_chebyshev, cZ[layer][neurona], -10, 10, 15); // KONTUZ!! tartea eta gradua

            /*if(depth - cA[layer][neurona]->GetLevel() < 10 && cA[layer][neurona]->GetLevel() > 0){
                cA[layer][neurona] = cc->EvalBootstrap(cA[layer][neurona]);
            }*/
            // dSigmoid
            cZ[layer][neurona] = cc->EvalMult(cA[layer][neurona], cc->EvalSub(pBat, cA[layer][neurona]));

            if(depth - cA[layer][neurona]->GetLevel() < 8){ // Merge egiteko
                cA[layer][neurona] = cc->EvalBootstrap(cA[layer][neurona]);
            }

            if(depth - cZ[layer][neurona]->GetLevel() < 6){
                cZ[layer][neurona] = cc->EvalBootstrap(cZ[layer][neurona]);
            }

        }

        if(cA[layer+1].size() == 1) cY = cA[layer+1][0]; // baldin bitarra, else merge? ALDATU

        if(depth - cY->GetLevel() < 4){
            cY = cc->EvalBootstrap(cY);
        }
    }

    return cY; // prediction
}

Ciphertext<DCRTPoly> mse(CryptoContext<DCRTPoly>& cc,
                         std::vector<Ciphertext<DCRTPoly>>& cA,
                         std::vector<Ciphertext<DCRTPoly>>& cY,
                         int length){
    Ciphertext<DCRTPoly> c1;
    for(int i=0; i<length; i++){
        Ciphertext<DCRTPoly> cSub = cc->EvalSub(cA[i], cY[i]);
        Ciphertext<DCRTPoly> cMult = cc->EvalMult(cSub, cSub);
        if (i==0) c1 = cMult;
        else c1 = cc->EvalAdd(c1, cMult);
    }
    return cc->EvalMult(c1, 1.0/length);
}

Ciphertext<DCRTPoly> dmse(CryptoContext<DCRTPoly> &cc,
                          Ciphertext<DCRTPoly> &cA,
                          Ciphertext<DCRTPoly> &cY,
                          int length){

    Ciphertext<DCRTPoly> cSub = cc->EvalSub(cA, cY);
    return cc->EvalMult(cSub, 2.0/length);
}

Ciphertext<DCRTPoly> back_propagation(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    const std::vector<Ciphertext<DCRTPoly>>& cx,
    int input_size,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cA,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cZ,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cE,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cD,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cd,
    std::vector<Ciphertext<DCRTPoly>> &cy
    ){

    int hardware_threads = std::thread::hardware_concurrency();
    int omp_threads = omp_get_max_threads();

    int extra_threads = hardware_threads - omp_threads;
    int available_threads = std::max(1, extra_threads);

    // Loss
    int L = nn->n_layers-2;

    Ciphertext<DCRTPoly> loss = mse(cc, cA[L+1], cy, nn->layers_size[L+1]);

    // E last layer
    //std::cout << "\t\tLast Layer" << std::endl;
    #pragma omp parallel for num_threads(available_threads)
    for(int j = 0; j < nn->layers_size[L+1]; j++){

        cE[L][j] = dmse(cc, cA[L+1][j], cy[j], nn->layers_size[L+1]); // dLoss
        cE[L][j] = cc->EvalMult(cE[L][j], cZ[L+1][j]);
        cd[L][j] = cc->EvalAdd(cd[L][j], cE[L][j]);

        for(int k = 0; k < nn->layers_size[L]; k++) {
            cD[L][j][k] = cc->EvalAdd(cD[L][j][k], cc->EvalMult(cE[L][j], cA[L][k]));
        }
    }

    for(int l=L-1; l >= 0; l--){
        //std::cout << "\t\tLayer: " << layer << std::endl;
        #pragma omp parallel for num_threads(available_threads)
        for(int j = 0; j < nn->layers_size[l+1]; j++){
            //std::cout << "\t\t\tNeurona: " << j << std::endl;
            Ciphertext<DCRTPoly> c1;
            for(int k=0; k<nn->layers_size[l+2]; k++){
                if(k == 0) c1 = cc->EvalMult(cE[l+1][k], cW[l+1][k][j]);
                else c1 = cc->EvalAdd(c1, cc->EvalMult(cE[l+1][k], cW[l+1][k][j]));
            }

            cE[l][j] = cc->EvalMult(c1, cZ[l+1][j]);

            for(int k = 0; k < nn->layers_size[l]; k++) {
                cD[l][j][k] = cc->EvalAdd(cD[l][j][k], cc->EvalMult(cE[l][j], cA[l][k]));  // wljk eguneratzeko
            }

            cd[l][j] = cc->EvalAdd(cd[l][j], cE[l][j]); // blj eguneratzeko
        }
    }

    return loss;
}

void update(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cD,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cd,
    double lr,
    int batch_size // = 1
    ){

    int hardware_threads = std::thread::hardware_concurrency();
    int omp_threads = omp_get_max_threads();

    int extra_threads = hardware_threads - omp_threads;
    int available_threads = std::max(1, extra_threads);

    for(int layer = 0; layer < nn->n_layers - 1; layer++){
        //std::cout << "\t\tLayer: " << layer << std::endl;
        //#pragma omp parallel for num_threads(available_threads)
        for(int neurona = 0; neurona < nn->layers_size[layer+1]; neurona++){
            #pragma omp parallel for num_threads(available_threads)
            for(int k=0; k<nn->layers_size[layer]; k++){
                if(depth - cD[layer][neurona][k]->GetLevel() < 6)
                    cD[layer][neurona][k] = cc->EvalBootstrap(cD[layer][neurona][k]);

                if(depth - cW[layer][neurona][k]->GetLevel() < 6)
                    cW[layer][neurona][k] = cc->EvalBootstrap(cW[layer][neurona][k]);
                cW[layer][neurona][k] = cc->EvalSub(cW[layer][neurona][k], cc->EvalMult(cD[layer][neurona][k], cc->MakeCKKSPackedPlaintext(std::vector<double>{lr/batch_size}, 1)));
            }
            #pragma omp parallel for num_threads(available_threads)
            for(int k=0; k<nn->layers_size[layer]; k++){
                if(depth - cW[layer][neurona][k]->GetLevel() < 6){
                    cW[layer][neurona][k] = cc->EvalBootstrap(cW[layer][neurona][k]);
                }
            }
            if(depth - cd[layer][neurona]->GetLevel() < 6){
                cd[layer][neurona] = cc->EvalBootstrap(cd[layer][neurona]);
            }
            if(depth - cB[layer][neurona]->GetLevel() < 6){
                cB[layer][neurona] = cc->EvalBootstrap(cB[layer][neurona]);
            }

            cB[layer][neurona] = cc->EvalSub(cB[layer][neurona], cc->EvalMult(cd[layer][neurona], cc->MakeCKKSPackedPlaintext(std::vector<double>{lr/batch_size}, 1)));

            if(depth - cB[layer][neurona]->GetLevel() < 6){
                cB[layer][neurona] = cc->EvalBootstrap(cB[layer][neurona]);
            }
        }
    }
}
