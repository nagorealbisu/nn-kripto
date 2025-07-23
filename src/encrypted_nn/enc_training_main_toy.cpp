/*
  Simple examples for CKKS
 */

#define PROFILE

#include "openfhe.h"
#include <random>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "enc_train_functions.hpp"
#include "nn_crypto_context.hpp"
#include "ckks_utils.hpp"

#include "globals.hpp"

extern "C" {
    #include "nn.h"
    #include "train.h"
}

using namespace lbcrypto;

// batches = 1 hasteko
extern "C" int encrypted_dataset_training(nn_t *nn, ds_t *ds, int epochs, int batches, double lr){ // ALDATU:  dataset-a zifratuta pasatu zuzenean

    omp_set_num_threads(8);

    std::chrono::high_resolution_clock::time_point denb;

    // proba
    numSlots = ds->n_inputs; // sarearen dimentsioen arabera

    // CryptoContext-a hasieratu
    std::cout << "CryptoContext-a sortzen... " << std::endl;
    CryptoContext<DCRTPoly> cc;
        denb = std::chrono::high_resolution_clock::now();
    KeyPair<DCRTPoly> keys = init_crypto_context_ckks_train(cc, nn, numSlots);
        std::cout << "CryptoContext sortzeko Denbora = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - denb).count() << " ms" << std::endl;

    std::cout << "\nnumSlots = " << numSlots << std::endl;

    std::cout << "nn->n_layers = " << nn->n_layers << std::endl;

    int N = ds->n_samples; // input kop
    int input_size = ds->n_inputs; // bitarra --> input_size = 2
    std::cout << "N = " << N << ", input_size = " << input_size << std::endl;

    std::cout << "\nds->n_inputs = " << ds->n_inputs << std::endl;
    std::cout << "\nds->n_outputs = " << ds->n_outputs << std::endl;

    //std::vector<std::vector<double>> x(N, std::vector<double>(ds->n_inputs));
    //std::vector<std::vector<double>> y(N, std::vector<double>(ds->n_outputs));

    std::vector<double> x(ds->n_inputs);
    std::vector<double> y(ds->n_outputs);

    /*for(int i=0; i<N; i++){
        for(int j=0; j<input_size; j++){
            x[i][j] = ds->inputs[i*ds->n_inputs + j];
            //std::cout << "\nx[i][j]: " << x[i][j] << std::endl;
        }
    }
    for(int i=0; i<N; i++){
        for(int j=0; j<ds->n_outputs; j++){
            y[i][j] = ds->outputs[i*ds->n_outputs + j];
            //std::cout << "\ny[i][j]: " << y[i][j] << std::endl;
        }
    }*/

    //std::cout << "\nPacking plaintext..." << std::endl;

    // Make packed plaintext
    //    denb = std::chrono::high_resolution_clock::now();
    //std::vector<std::vector<Plaintext>> px(N, std::vector<Plaintext>(ds->n_inputs)), py(N, std::vector<Plaintext>(ds->n_outputs));
    /*for(int i = 0; i < N; i++){
        for(int j = 0; j < ds->n_inputs; j++){
            px[i][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{x[i][j]});
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < ds->n_outputs; j++){
            py[i][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{y[i][j]});
        }
    }*/

    std::vector<std::vector<std::vector<Plaintext>>> pW(nn->n_layers - 1), pD(nn->n_layers - 1);
    std::vector<std::vector<Plaintext>> pA(nn->n_layers), pZ(nn->n_layers),
                                        pB(nn->n_layers - 1), pE(nn->n_layers - 1),
                                        pd(nn->n_layers);

    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>> cW(nn->n_layers - 1), cD(nn->n_layers - 1);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> cA(nn->n_layers), cZ(nn->n_layers),
                                                   cB(nn->n_layers - 1), cE(nn->n_layers - 1),
                                                   cd(nn->n_layers - 1);

    std::vector<double> zero = {0.0};
    Plaintext pZero = cc->MakeCKKSPackedPlaintext(zero);
    Ciphertext<DCRTPoly> cZero = cc->Encrypt(keys.publicKey, pZero);

    for(int layer = 0; layer < nn->n_layers; layer++) {
        int size = nn->layers_size[layer];

        pA[layer].resize(size, pZero);
        pZ[layer].resize(size, pZero);

        cA[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));
        cZ[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));

        if(layer < nn->n_layers - 1) {
            pd[layer].resize(nn->layers_size[layer + 1], pZero);
            pE[layer].resize(nn->layers_size[layer + 1], pZero);
            cd[layer].resize(nn->layers_size[layer + 1], cc->Encrypt(keys.publicKey, pZero));
            cE[layer].resize(nn->layers_size[layer + 1], cc->Encrypt(keys.publicKey, pZero));
        }
    }

    init_zero(cA, cZero);
    init_zero(cZ, cZero);
    init_zero(cd, cZero);
    init_zero(cE, cZero);

    std::vector<double> batbat = {1.0};
    Plaintext pBatbat = cc->MakeCKKSPackedPlaintext(batbat);
    Ciphertext<DCRTPoly> cBatbat = cc->Encrypt(keys.publicKey, pBatbat);
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> cW(nn->n_layers-1);

    for(int layer = 0; layer < nn->n_layers - 1; layer++) {
        int rows = nn->layers_size[layer];
        int cols = nn->layers_size[layer + 1];

        cW[layer].resize(cols);
        cD[layer].resize(cols);
        pW[layer].resize(cols);
        pD[layer].resize(cols);

        for(int j = 0; j < cols; j++) {
            cW[layer][j].resize(rows);
            cD[layer][j].resize(rows);
            pW[layer][j].resize(rows);
            pD[layer][j].resize(rows);

            for(int k = 0; k < rows; k++) {
                double w = nn->WH[layer][j * rows + k];
                pW[layer][j][k] = cc->MakeCKKSPackedPlaintext(std::vector<double>{w});
                cW[layer][j][k] = cc->EvalMult(cBatbat, pW[layer][j][k]);
            }
        }

        pB[layer].resize(cols);
        cB[layer].resize(cols);
        for(int j = 0; j < cols; j++) {
            double b = nn->BH[layer][j];
            pB[layer][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{b});
            cB[layer][j] = cc->EvalMult(cBatbat, pB[layer][j]);
        }
    }

    init_zero(cd, cZero);
    init_zero_3D(cD, cZero);

    //std::cout << "\nEncrypting..." << std::endl;

    // Encrypt
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> cx(N, std::vector<Ciphertext<DCRTPoly>>(input_size)), cy(N, std::vector<Ciphertext<DCRTPoly>>(ds->n_outputs));
    /*for(int i = 0; i < N; i++){
        for(int j = 0; j < input_size; j++){
            cx[i][j] = cc->Encrypt(keys.publicKey, px[i][j]);
        }
        for(int j = 0; j < ds->n_outputs; j++){
            cy[i][j] = cc->Encrypt(keys.publicKey, py[i][j]);
        }
    }*/

    //std::cout << "\nMakePackedPlaintext eta Encrypt egiteko Denbora totala = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - denb).count() << " ms" << std::endl;

    std::vector<Ciphertext<DCRTPoly>> cY(N); // predictions (y)

    // Denbora
    std::chrono::high_resolution_clock::time_point start, end, t_epoch;

    double loss_totala = 0.0;

    std::cout << "\nStarting training phase..." << std::endl;

    std::vector<Plaintext> px(ds->n_inputs), py(ds->n_outputs);
    std::vector<Ciphertext<DCRTPoly>> cx(ds->n_inputs), cy(ds->n_outputs);

    for(int epoch = 0; epoch<epochs; epoch++){
        std::cout << "\nEPOCH " << epoch << std::endl;
        t_epoch = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < N; i++){
            //std::vector<Plaintext> px(ds->n_inputs);
            for(int j = 0; j < input_size; j++){
                x[j] = ds->inputs[i*ds->n_inputs + j];
                px[j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{x[j]});
                cx[j] = cc->Encrypt(keys.publicKey, px[j]);
            }
            for(int j = 0; j < ds->n_outputs; j++){
                y[j] = ds->outputs[i*ds->n_outputs + j];
                py[j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{y[j]});
                cy[j] = cc->Encrypt(keys.publicKey, py[j]);
            }

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Input: " << i;
            //std::cout << "\n\nFORDWARD PASS\n" << std::endl;
            cY[i] = forward_pass(cc, nn, cW, cB, cx, input_size, cA, cZ);

            //std::cout << "\nBACK PROPAGATION" << std::endl;
            Ciphertext<DCRTPoly> cLoss = back_propagation(cc, nn, cW, cB, cx, input_size, cA, cZ, cE, cD, cd, cy);

            //std::cout << "\nUPDATE" << std::endl;
            update(cc, nn, cW, cB, cD, cd, lr, batches);

            init_zero(cE, cZero);
            init_zero(cd, cZero);
            init_zero_3D(cD, cZero);

            end = std::chrono::high_resolution_clock::now();
            std::cout << ", Denbora = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

            Plaintext pLoss;
            cc->Decrypt(keys.secretKey, cLoss, &pLoss);
            pLoss->SetLength(1);
            double l = pLoss->GetCKKSPackedValue()[0].real();
            //std::cout << "Loss: " << l << std::endl;
            loss_totala+=l;
            //print_net(cc, keys, nn, cW, cB);
        }
        std::cout << "Denbora Epoch = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - t_epoch).count() << " ms" << std::endl;
        std::cout << "\t\t\t\tLoss: " << loss_totala/N << std::endl;
        loss_totala = 0.0;
        std::cout.precision(8);
        print_net(cc, keys, nn, cW, cB);
    }

    return 0;
}

