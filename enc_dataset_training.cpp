/*
  Simple examples for CKKS
 */

#define PROFILE

#include "openfhe.h"
#include <random>
#include <cmath>
#include <iostream>

extern "C" {
    #include "nn.h"
    #include "train.h"
}

// Sigmoid: 1 / (1 + exp(-x))
auto sigmoid_chebyshev = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
};

using namespace lbcrypto;

uint32_t depth;
uint32_t numSlots;

void print_net(CryptoContext<DCRTPoly>& cc,
    KeyPair<DCRTPoly>& keys, 
    nn_t* nn, 
    const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB){

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Network:" << std::endl;
    for(int layer=0; layer<nn->n_layers-1; layer++){
        std::cout << "Layer " << layer << " Weights:" << std::endl;
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            std::cout << "Neuron " << neuron << ": [";
            for(int weight=0; weight<nn->layers_size[layer]; weight++){
                Plaintext pW;
                cc->Decrypt(keys.secretKey, cW[layer][neuron][weight], &pW);
                pW->SetLength(1);
                std::cout << pW->GetCKKSPackedValue()[0].real() << " ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Layer " << layer << " Biases:" << std::endl;
        std::cout << "[";
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            Plaintext pB;
            cc->Decrypt(keys.secretKey, cB[layer][neuron], &pB);
            pB->SetLength(1);
            std::cout << pB->GetCKKSPackedValue()[0].real() << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
}

void decrypt_net(CryptoContext<DCRTPoly>& cc,
    KeyPair<DCRTPoly>& keys, 
    nn_t* nn, 
    const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    std::vector<std::vector<std::vector<Plaintext>>>& pW,
    std::vector<std::vector<Plaintext>>& pB
    ){

    for(int layer=0; layer<nn->n_layers-1; layer++){
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            for(int weight=0; weight<nn->layers_size[layer]; weight++){
                cc->Decrypt(keys.secretKey, cW[layer][neuron][weight], &pW[layer][neuron][weight]);
                pW[layer][neuron][weight]->SetLength(1);
            }
        }
        
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            cc->Decrypt(keys.secretKey, cB[layer][neuron], &pB[layer][neuron]);
            pB[layer][neuron]->SetLength(1);
        }
    }
}

void init_zero(std::vector<std::vector<Ciphertext<DCRTPoly>>>& cM, const Ciphertext<DCRTPoly>& cZero){
    for(auto &layer: cM){
        for(auto &x: layer){
            x = cZero;
        }
    }
}

void init_zero_3D(std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cM, 
                 const Ciphertext<DCRTPoly>& cZero){
    for(auto &layer: cM){
        for(auto& neuron: layer){
            for(auto &x: neuron){
                x = cZero;
            }
        }
    }
}

void print_matrix(std::vector<std::vector<Ciphertext<DCRTPoly>>>& cM,
    KeyPair<DCRTPoly> keys, CryptoContext<DCRTPoly> &cc){
    for(auto &layer: cM){
        std::cout << std::endl;
        for(auto &bal: layer){
            Plaintext pB;
            cc->Decrypt(keys.secretKey, bal, &pB);
            pB->SetLength(1);
            std::cout << pB->GetCKKSPackedValue()[0].real() << " ";
        }
    }
}

void print_matrix_3D(std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cM,
    KeyPair<DCRTPoly> keys, CryptoContext<DCRTPoly> &cc){
    for(auto &layer: cM){
        std::cout << std::endl;
        for(auto &neuron: layer){
            for(auto &x: neuron){
                Plaintext pB;
                cc->Decrypt(keys.secretKey, x, &pB);
                pB->SetLength(1);
                std::cout << pB->GetCKKSPackedValue()[0].real() << " ";
            }
        }
    }
}

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

    Ciphertext<DCRTPoly> cY = cc->EvalMerge(cx);

    for(int i=0; i<input_size; i++){
        cA[0][i] = cx[i];
    }

    std::vector<double> bat = {1.0};
    Plaintext pBat = cc->MakeCKKSPackedPlaintext(bat);

    for (int layer = 1; layer < nn->n_layers; layer++){
        //std::cout << "Layer: " << layer << std::endl;
        for(int neurona = 0; neurona < nn->layers_size[layer]; neurona++){
          
            for(size_t k=0; k<cW[layer-1][neurona].size(); k++)
                if(depth - cW[layer-1][neurona][k]->GetLevel() < 6)
                    cW[layer-1][neurona][k] = cc->EvalBootstrap(cW[layer-1][neurona][k]);
            
            // wx + b
            cZ[layer][neurona] = cc->EvalAdd(cc->EvalInnerProduct(cY, cc->EvalMerge(cW[layer-1][neurona]), nn->layers_size[layer]), cB[layer-1][neurona]);

            if(depth - cZ[layer][neurona]->GetLevel() < 8){
                cZ[layer][neurona] = cc->EvalBootstrap(cZ[layer][neurona]);
            }

            // Sigmoid
            cA[layer][neurona] = cc->EvalChebyshevFunction(sigmoid_chebyshev, cZ[layer][neurona], -6, 6, 3); // KONTUZ!! tartea eta gradua

            if(depth - cA[layer][neurona]->GetLevel() < 10 && cA[layer][neurona]->GetLevel() > 0){
                cA[layer][neurona] = cc->EvalBootstrap(cA[layer][neurona]);
            }
            // dSigmoid
            cZ[layer][neurona] = cc->EvalMult(cA[layer][neurona], cc->EvalSub(pBat, cA[layer][neurona]));
            if(depth - cA[layer][neurona]->GetLevel() < 8){ // Merge egiteko
                cA[layer][neurona] = cc->EvalBootstrap(cA[layer][neurona]);
            }
            if(depth - cZ[layer][neurona]->GetLevel() < 6){
                cZ[layer][neurona] = cc->EvalBootstrap(cZ[layer][neurona]);
            }
        }

        if(cA[layer].size() > 1) cY = cc->EvalMerge(cA[layer]);
        else cY = cA[layer][0];
        
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
    for (int i=0; i<length; i++){
        if(depth - cA[i]->GetLevel() < 6){
            cA[i] = cc->EvalBootstrap(cA[i]);
        }
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

    // Loss
    int L = nn->n_layers-2;

    Ciphertext<DCRTPoly> loss = mse(cc, cA[L+1], cy, nn->layers_size[L+1]);

    // E last layer
    //std::cout << "Last Layer" << std::endl;
    for(int j = 0; j < nn->layers_size[L+1]; j++){

        cE[L][j] = dmse(cc, cA[L+1][j], cy[j], 1); // dLoss
        cE[L][j] = cc->EvalMult(cE[L][j], cZ[L+1][j]);
        cd[L][j] = cc->EvalAdd(cd[L][j], cE[L][j]);

        for(int k = 0; k < nn->layers_size[L]; k++) {
            cD[L][j][k] = cc->EvalAdd(cD[L][j][k], cc->EvalMult(cE[L][j], cA[L][k]));
        }
    }
    //std::cout << "Hidden Layers" << std::endl;
    for (int l = L-1; l >= 0; l--){
        //std::cout << "Layer: " << l << std::endl;
        for(int j = 0; j < nn->layers_size[l+1]; j++){
            //std::cout << "Neurona: " << j << std::endl;
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

    for (int layer = 0; layer < nn->n_layers - 1; layer++){
        for(int neurona = 0; neurona < nn->layers_size[layer+1]; neurona++){

            for(int k=0; k<nn->layers_size[layer]; k++){
                if(depth - cW[layer][neurona][k]->GetLevel() < 6){
                    cW[layer][neurona][k] = cc->EvalBootstrap(cW[layer][neurona][k]);
                }
                for(Ciphertext<DCRTPoly> &k: cD[layer][neurona])
                    if(depth - k->GetLevel() < 6)
                        k = cc->EvalBootstrap(k);
                
                if(depth - cW[layer][neurona][k]->GetLevel() < 6)
                    cW[layer][neurona][k] = cc->EvalBootstrap(cW[layer][neurona][k]);
                
                cW[layer][neurona][k] = cc->EvalSub(cW[layer][neurona][k], cc->EvalMult(cD[layer][neurona][k], lr/batch_size));
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

            cB[layer][neurona] = cc->EvalSub(cB[layer][neurona], cc->EvalMult(cd[layer][neurona], lr/batch_size));
            if(depth - cB[layer][neurona]->GetLevel() < 6){
                cB[layer][neurona] = cc->EvalBootstrap(cB[layer][neurona]);
            }
        }
    }
}

KeyPair<DCRTPoly> init_crypto_context_ckks_train(CryptoContext<DCRTPoly> &cc, nn_t *nn, uint32_t &numSlots){
    //uint32_t multDepth = 16; // xor 12; diabetes 12, 18;
    //uint32_t scaleModSize = 40; // xor 22; diabetes 40, 50;
    CCParams<CryptoContextCKKSRNS> parameters;

    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);  // jostailuzkoa --> HEStd_128_classic gutxienez
    parameters.SetRingDim(1 << 12); // jostailuzkoa. HEStd_128_classic --> 1<<17 gutxienez

    #if NATIVEINT == 128
        parameters.SetScalingTechnique(FIXEDAUTO);
        parameters.SetScalingModSize(78);
        parameters.SetFirstModSize(89);
    #else
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);
    #endif

    std::vector<uint32_t> levelBudget = {2, 2}; // edo {2,2}. {1,1} errorea bootstrapping egiterakoan
    std::vector<uint32_t> bsgsDim = {0, 0};

    uint32_t levelsAfterBootstrap = 4; // 8 ondo
    depth = levelsAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    std::cout << "\nDepth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);

    //numSlots = 8;
    parameters.SetBatchSize(numSlots); // neurona kopuruarena arabera --> XOR, OR, ...: 2

    cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeysGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    //numSlots = cc->GetRingDimension() / 2;

    // EvalMerge erabiltzeko beharrezkoak diren RotationKey-ak sortu
    std::vector<int32_t> indexList;

    for(int i = 1; i < (int)numSlots; i++){ // ALDATUTA
        indexList.push_back(i);
        indexList.push_back(-i);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, indexList);

    // Bootstrapping keys
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
    //parameters.SetBatchSize(numSlots);

    return keys;
}

// batches = 1 hasteko
extern "C" int encrypted_dataset_training(nn_t *nn, ds_t *ds, int epochs, int batches, double lr){

    // proba
    numSlots = ds->n_inputs; // sarearen dimentsioen arabera

    // CryptoContext-a hasieratu
    std::cout << "CryptoContext sortzen... " << std::endl;
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys = init_crypto_context_ckks_train(cc, nn, numSlots);

    //std::cout << "\nnumSlots = " << numSlots << std::endl;
    //std::cout << "nn->n_layers = " << nn->n_layers << std::endl;

    int N = ds->n_samples;
    int input_size = ds->n_inputs; // bitarra --> input_size = 2
    std::cout << "N = " << N << ", input_size = " << input_size << std::endl;

    std::cout << "\nds->n_inputs = " << ds->n_inputs << std::endl;
    std::cout << "\nds->n_outputs = " << ds->n_outputs << std::endl;

    std::vector<std::vector<double>> x(N, std::vector<double>(ds->n_inputs));
    std::vector<std::vector<double>> y(N, std::vector<double>(ds->n_outputs));
    for(int i=0; i<N; i++){
        for(int j=0; j<input_size; j++){
            x[i][j] = ds->inputs[i*ds->n_inputs + j];
            //std::cout << "x[i][j]" << x[i][j] << std::endl;
        }
        for(int j=0; j<ds->n_outputs; j++){
            y[i][j] = ds->outputs[(ds->n_inputs * i) + j];
        }
    }

    std::cout << "\nPacking plaintext..." << std::endl;

    // Make packed plaintext
    std::vector<std::vector<Plaintext>> px(N, std::vector<Plaintext>(ds->n_inputs)), py(N, std::vector<Plaintext>(ds->n_outputs));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < ds->n_inputs; j++){
            px[i][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{x[i][j]});
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < ds->n_outputs; j++){
            py[i][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{y[i][j]});
        }
    }

    std::vector<std::vector<std::vector<Plaintext>>> pW(nn->n_layers - 1), pD(nn->n_layers - 1);
    std::vector<std::vector<Plaintext>> pA(nn->n_layers), pZ(nn->n_layers),
                                        pB(nn->n_layers - 1), pE(nn->n_layers),
                                        pd(nn->n_layers);

    
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>> cW(nn->n_layers - 1), cD(nn->n_layers - 1);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> cA(nn->n_layers), cZ(nn->n_layers),
                                                cB(nn->n_layers - 1), cE(nn->n_layers-1),
                                                cd(nn->n_layers - 1);

    std::vector<double> zero = {0.0};
    Plaintext pZero = cc->MakeCKKSPackedPlaintext(zero);
    Ciphertext<DCRTPoly> cZero = cc->Encrypt(keys.publicKey, pZero);

    for(int layer = 0; layer < nn->n_layers; layer++){
        int size = nn->layers_size[layer];
        
        pA[layer].resize(size, pZero);
        pZ[layer].resize(size, pZero);

        cA[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));
        cZ[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));
        
        if(layer < nn->n_layers-1){
            pd[layer].resize(size, pZero);
            pE[layer].resize(size, pZero);
            cd[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));
            cE[layer].resize(size, cc->Encrypt(keys.publicKey, pZero));
        }
    }

    init_zero(cA, cZero);
    init_zero(cZ, cZero);
    init_zero(cd, cZero);
    init_zero(cE, cZero);

    std::vector<double> batbat = {1.0};
    Plaintext pBatbat = cc->MakeCKKSPackedPlaintext(batbat);
    Ciphertext<DCRTPoly> cBatbat = cc->Encrypt(keys.publicKey, pBatbat);

    for(int layer = 0; layer < nn->n_layers - 1; layer++){
        int rows = nn->layers_size[layer];
        int cols = nn->layers_size[layer+1];
        
        cW[layer].resize(cols);
        cD[layer].resize(cols);
        pW[layer].resize(cols);
        pD[layer].resize(cols);
        
        for(int j = 0; j < cols; j++){
            cW[layer][j].resize(rows);
            cD[layer][j].resize(rows);
            pW[layer][j].resize(rows);
            pD[layer][j].resize(rows);
            
            for(int k = 0; k < rows; k++){
                double w = nn->WH[layer][j*rows + k];
                pW[layer][j][k] = cc->MakeCKKSPackedPlaintext(std::vector<double>{w});
                cW[layer][j][k] = cc->EvalMult(cBatbat, pW[layer][j][k]);
            }
        }
        
        pB[layer].resize(cols);
        cB[layer].resize(cols);
        for(int j = 0; j < cols; j++){
            double b = nn->BH[layer][j];
            pB[layer][j] = cc->MakeCKKSPackedPlaintext(std::vector<double>{b});
            cB[layer][j] = cc->EvalMult(cBatbat, pB[layer][j]);
        }
    }

    init_zero(cd, cZero);
    init_zero_3D(cD, cZero);

    std::cout << "\nEncrypting..." << std::endl;

    // Encrypt
    std::vector<std::vector<Ciphertext<DCRTPoly>>> cx(N, std::vector<Ciphertext<DCRTPoly>>(input_size)), cy(N, std::vector<Ciphertext<DCRTPoly>>(ds->n_outputs));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < input_size; j++){
            cx[i][j] = cc->Encrypt(keys.publicKey, px[i][j]);
        }
        for(int j = 0; j < ds->n_outputs; j++){
            cy[i][j] = cc->Encrypt(keys.publicKey, py[i][j]);
        }
    }

    std::vector<Ciphertext<DCRTPoly>> cY(N); // predictions (y)

    // Denbora
    std::chrono::high_resolution_clock::time_point start, end, t_epoch;

    double loss_totala = 0.0;

    std::cout << "\nStarting training phase..." << std::endl;

    for(int epoch = 0; epoch<15; epoch++){
        std::cout << "\nEpoch " << epoch << std::endl;
        t_epoch = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < N; i++){
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Input: " << i;
            //std::cout << "\nFORDWARD PASS\n" << std::endl;
            cY[i] = forward_pass(cc, nn, cW, cB, cx[i], input_size, cA, cZ);
            //std::cout << "\nBACK PROPAGATION" << std::endl;
            Ciphertext<DCRTPoly> cLoss = back_propagation(cc, nn, cW, cB, cx[i], input_size, cA, cZ, cE, cD, cd, cy[i]);
            //std::cout << "\nUPDATE" << std::endl;*/
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
        }
        std::cout << "Denbora Epoch = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - t_epoch).count() << " ms" << std::endl;
        std::cout << "\t\t\t\tLoss: " << loss_totala/N << std::endl;
        loss_totala = 0.0;
        print_net(cc, keys, nn, cW, cB);
    }

    decrypt_net(cc, keys, nn, cW, cB, pW, pB);

    for(int layer = 0; layer < nn->n_layers-1; layer++){
        int rows = nn->layers_size[layer];
        int cols = nn->layers_size[layer+1];
        
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < rows; k++){
                nn->WH[layer][j*rows + k] = pW[layer][j][k]->GetCKKSPackedValue()[0].real();
            }
        }
        
        for(int j = 0; j < cols; j++){
            nn->BH[layer][j] = pB[layer][j]->GetCKKSPackedValue()[0].real();
        }
    }

    return 0;
}
