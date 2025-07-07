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
    #include "test.h"
}

// Sigmoid: 1 / (1 + exp(-x))
auto sigmoid_chebyshev_test = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
};

using namespace lbcrypto;

uint32_t depth;

Ciphertext<DCRTPoly> forward_pass(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    const std::vector<std::vector<Plaintext>>& pW,
    const std::vector<std::vector<Plaintext>>& pB,
    const Ciphertext<DCRTPoly>& cx,
    int input_size){

    Ciphertext<DCRTPoly> cY = cx;

    std::chrono::high_resolution_clock::time_point start, end, t1, t2, t3;

    start = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < nn->n_layers - 1; layer++){
        std::vector<Ciphertext<DCRTPoly>> cNeuronak(nn->layers_size[layer+1]);
        std::cout << "Layer " << layer << ", ciphertext level: " << cY->GetLevel() << std::endl;
        // paralelizatu
        for(int neurona = 0; neurona < nn->layers_size[layer+1]; neurona++){
            t1 = std::chrono::high_resolution_clock::now();

            cNeuronak[neurona] = cc->EvalAdd(cc->EvalInnerProduct(cY, pW[layer][neurona], input_size), pB[layer][neurona]);
            t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Denbora wx + b " << neurona << ". neurona: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
            
            std::cout << "Level after inner product: " << cNeuronak[neurona]->GetLevel() << std::endl;

            cNeuronak[neurona] = cc->EvalChebyshevFunction(sigmoid_chebyshev_test, cNeuronak[neurona], -10, 10, 5);
            t3 = std::chrono::high_resolution_clock::now();
            std::cout << "Denbora chebyshev " << neurona << ". neurona: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;
            
            std::cout << "Level after Chebyshev: " << cNeuronak[neurona]->GetLevel() << std::endl;

            std::cout << "Denbora totala " << neurona << ". neurona: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count() << " ms\n" << std::endl;
        }
        // sarraila
        std::cout << "EvalMerge..." << std::endl;
        cY = cc->EvalMerge(cNeuronak);

        if(depth - cY->GetLevel() < 4){ // ONDO PENTSATU: sarearen dimentsioen arabera konfiguratu??
            std::cout << "Bootstrapping egiten saiatzen..." << std::endl;
            t1 = std::chrono::high_resolution_clock::now();
            cY = cc->EvalBootstrap(cY);
            t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Denbora Bootstrapping: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
        }
        
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Denbora Totala: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n" << std::endl;

    return cY; // prediction
}

KeyPair<DCRTPoly> init_crypto_context_ckks(CryptoContext<DCRTPoly> &cc, nn_t *nn, uint32_t &numSlots){
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

    std::vector<uint32_t> levelBudget = {2, 2}; // edo {3,3}. {1,1} errorea bootstrapping egiterakoan
    std::vector<uint32_t> bsgsDim = {0, 0};

    uint32_t levelsAfterBootstrap = 2;
    depth = levelsAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    std::cout << "\nDepth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);

    numSlots = 8;
    parameters.SetBatchSize(numSlots);

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
    /*int max = 0;
    for(int i = 0; i < nn->n_layers; i++){
        if (nn->layers_size[i] > max)
            max = nn->layers_size[i];
    }*/
    for(int i = 1; i < (int)numSlots; i++){ // ALDATUTA
        indexList.push_back(i);
        indexList.push_back(-i);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, indexList);

    // Bootstrapping keys
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    return keys;
}

extern "C" int encrypted_dataset_testing(nn_t *nn, ds_t *ds) {

    // proba
    uint32_t numSlots = 8; // sarearen dimentsioen arabera

    // CryptoContext-a hasieratu
    std::cout << "CryptoContext-a sortzen... " << std::endl;
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys = init_crypto_context_ckks(cc, nn, numSlots);

    std::cout << "\nnumSlots = " << numSlots << std::endl;
    
    int N = ds->n_samples; // input kop
    int input_size = nn->layers_size[0]; // bitarra --> input_size = 2
    std::cout << "N = " << N << ", input_size = " << input_size << std::endl;

    std::vector<std::vector<double>> x(N, std::vector<double>(numSlots, 0.0));
    for(int i=0; i<N; i++){
        for(int j=0; j<input_size; j++){
            x[i][j] = ds->inputs[i*ds->n_inputs + j];
            //std::cout << "x[i][j]" << x[i][j] << std::endl;
        }
    }

    // Make packed plaintext
    std::vector<Plaintext> px(N);
    for(int i = 0; i < N; i++){
        //px[i] = cc->MakeCKKSPackedPlaintext(x[i], 1, depth - 1, nullptr, numSlots);
        px[i] = cc->MakeCKKSPackedPlaintext(x[i]);
    }

    std::vector<std::vector<Plaintext>> pW(nn->n_layers - 1);
    std::vector<std::vector<Plaintext>> pB(nn->n_layers - 1);
    //std::vector<Plaintext> pB(nn->n_layers - 1);

    // funtzio bat sortu: pack plaintext
    for (int i = 0; i < nn->n_layers - 1; i++) {
        int rows = nn->layers_size[i];
        int cols = nn->layers_size[i + 1];
        double* Wi = nn->WH[i];

        pW[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            std::vector<double> Wj(numSlots, 0.0);
            for (int k = 0; k < rows; k++) {
                Wj[k] = Wi[j * rows + k];
            }
            //pW[i][j] = cc->MakeCKKSPackedPlaintext(Wj, 1, depth - 1, nullptr, numSlots);
            pW[i][j] = cc->MakeCKKSPackedPlaintext(Wj);
        }

        std::vector<double> Bi(nn->BH[i], nn->BH[i] + cols);
        pB[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            std::vector<double> bj(numSlots, 0.0);
            bj[0] = Bi[j];
            //pB[i][j] = cc->MakeCKKSPackedPlaintext(bj, 1, depth - 1, nullptr, numSlots);
            pB[i][j] = cc->MakeCKKSPackedPlaintext(bj);
        }
    }

    // Encrypt
    std::vector<Ciphertext<DCRTPoly>> cx(N);
    for(int i = 0; i < N; i++) cx[i] = cc->Encrypt(keys.publicKey, px[i]);
    
    std::vector<Ciphertext<DCRTPoly>> cY(N); // predictions (y)

    // Denbora
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();

    // paralelizatu --> dummy bat sortu lehenengo
    for(int i = 0; i < N; i++) {
        cY[i] = forward_pass(cc, nn, pW, pB, cx[i], input_size);
        std::cout << i << std::endl;
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << "\nDenbora totala: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "BB denbora: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())/N << " ms" << std::endl;
    std::cout << "\nPredictions = \n";

    int tp = 0, fp = 0, tn = 0, fn = 0;

    for(int i = 0; i < N; i++) {
      Plaintext emaitza;
      cc->Decrypt(keys.secretKey, cY[i], &emaitza);
      emaitza->SetLength(1);
      double y = emaitza->GetCKKSPackedValue()[0].real();
      //std::cout << x[i][0] << " OR " << x[i][1] << " = " << (x[i][0] || x[i][1]);
      std::cout << "Prediction (real): " << emaitza->GetCKKSPackedValue()[0].real();
      int pred = (y >= 0.5) ? 1 : 0;
      std::cout << ", Prediction: " << pred;
      std::cout << ", Output: " << ds->outputs[i] << std::endl;

      if(ds->outputs[i] == 1){
        if(pred == 1) tp++;
        else fn++;
      }else{
        if(pred == 1) fp++;
        else tn++;
      }
    }

    std::cout << "TP: " << tp << ", FP: " << fp << std::endl;
    std::cout << "FN: " << fn << ", TN: " << tn << std::endl;

    double precision = 0.0, recall = 0.0, f1 = 0.0;
    if(tp + fp != 0){
        precision = static_cast<double>(tp) / (tp + fp);
        recall = static_cast<double>(tp) / (tp + fn);
    }
    if(precision + recall != 0) f1 = static_cast<double>(2*precision*recall)/(precision+recall);
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1: " << f1 << std::endl;

    return 0;
}
