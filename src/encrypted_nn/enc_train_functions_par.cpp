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
#include "ckks_utils.hpp"

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
            // wx + b
            cZ[layer][neurona] = cB[layer-1][neurona];
            std::vector<Ciphertext<DCRTPoly>> cMult(nn->layers_size[layer-1]), cMultRescale(nn->layers_size[layer-1]);

            #pragma omp simd
            for(int j=0; j<nn->layers_size[layer-1]; j++){
                cMult[j] = cc->EvalMult(cA[layer-1][j], cW[layer-1][neurona][j]);
                cMultRescale[j] = cc->Rescale(cMult[j]);
            }

            for(int j=0; j<nn->layers_size[layer-1]; j++){
                cZ[layer][neurona] = cc->EvalAdd(cZ[layer][neurona], cMultRescale[j]);
            }
            //bootstrap(cc, cZ[layer][neurona], 8);

            // Sigmoid
            cA[layer][neurona] = cc->EvalChebyshevFunction(sigmoid_chebyshev, cZ[layer][neurona], -10, 10, 15); // KONTUZ!! tartea eta gradua

            bootstrap(cc, cA[layer][neurona], 10);

            // dSigmoid
            auto cSub = cc->Rescale(cc->EvalSub(pBat, cA[layer][neurona]));
            cZ[layer][neurona] = cc->EvalMult(cA[layer][neurona], cSub);

            cZ[layer][neurona] = cc->Rescale(cZ[layer][neurona]);

        }

        if(cA[layer+1].size() == 1) cY = cA[layer+1][0]; // baldin bitarra

        /*if(depth - cY->GetLevel() < 4){
            cY = cc->EvalBootstrap(cY);
        }*/
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
    //#pragma omp parallel for num_threads(available_threads)
    #pragma omp parallel for schedule(static) num_threads(available_threads)
    for(int j = 0; j < nn->layers_size[L+1]; j++){

        cE[L][j] = dmse(cc, cA[L+1][j], cy[j], nn->layers_size[L+1]); // dLoss
        cE[L][j] = cc->EvalMult(cE[L][j], cZ[L+1][j]);
        cd[L][j] = cc->EvalAdd(cd[L][j], cE[L][j]);

        #pragma omp simd
        for(int k = 0; k < nn->layers_size[L]; k++) {
            cD[L][j][k] = cc->EvalAdd(cD[L][j][k], cc->EvalMult(cE[L][j], cA[L][k]));
        }
    }

    for(int l=L-1; l >= 0; l--){
        //std::cout << "\t\tLayer: " << layer << std::endl;
        //#pragma omp parallel for num_threads(available_threads)
        #pragma omp parallel for schedule(static) num_threads(available_threads)
        for(int j = 0; j < nn->layers_size[l+1]; j++){
            //std::cout << "\t\t\tNeurona: " << j << std::endl;
            Ciphertext<DCRTPoly> c1;
            for(int k=0; k<nn->layers_size[l+2]; k++){
                if(k == 0) c1 = cc->EvalMult(cE[l+1][k], cW[l+1][k][j]);
                else c1 = cc->EvalAdd(c1, cc->EvalMult(cE[l+1][k], cW[l+1][k][j]));
            }

            cE[l][j] = cc->EvalMult(c1, cZ[l+1][j]);

            #pragma omp simd
            for(int k = 0; k < nn->layers_size[l]; k++) {
                cD[l][j][k] = cc->EvalAdd(cD[l][j][k], cc->EvalMult(cE[l][j], cA[l][k]));  // wljk eguneratzeko
            }

            cd[l][j] = cc->EvalAdd(cd[l][j], cE[l][j]); // blj eguneratzeko
        }
    }

    return loss;
}

// paralelo
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
        for(int neurona = 0; neurona < nn->layers_size[layer+1]; neurona++){

            #pragma omp parallel for num_threads(available_threads)
            for(int k=0; k<nn->layers_size[layer]; k++){
                //bootstrap(cc, cD[layer][neurona][k], 6);
                //bootstrap(cc, cW[layer][neurona][k], 6);
                auto cMult = cc->Rescale(cc->EvalMult(cD[layer][neurona][k], cc->MakeCKKSPackedPlaintext(std::vector<double>{lr/batch_size}, 1)));
                auto cSub = cc->EvalSub(cW[layer][neurona][k], cMult);
                cW[layer][neurona][k] = cc->Rescale(cSub);
                //bootstrap(cc, cD[layer][neurona][k], 6);
            }

            //bootstrap(cc, cd[layer][neurona], 6);
            //bootstrap(cc, cB[layer][neurona], 6);
            auto cMult = cc->Rescale(cc->EvalMult(cd[layer][neurona], cc->MakeCKKSPackedPlaintext(std::vector<double>{lr/batch_size}, 1)));
            cB[layer][neurona] = cc->Rescale(cc->EvalSub(cB[layer][neurona], cMult));
            //bootstrap(cc, cB[layer][neurona], 6);
        }
    }
}
