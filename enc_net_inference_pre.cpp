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
auto sigmoid_chebyshev = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
};

using namespace lbcrypto;

uint32_t depth;

Ciphertext<DCRTPoly> forward_pass(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cW,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    const Plaintext& px,
    int input_size){

    Ciphertext<DCRTPoly> cY;

    for (int layer = 0; layer < nn->n_layers - 1; layer++){
        std::vector<Ciphertext<DCRTPoly>> cNeuronak(nn->layers_size[layer+1]);

        for(int neurona = 0; neurona < nn->layers_size[layer+1]; neurona++){
            
            if(layer == 0) cNeuronak[neurona] = cc->EvalAdd(cc->EvalInnerProduct(cW[layer][neurona], px, input_size), cB[layer][neurona]);
            else cNeuronak[neurona] = cc->EvalAdd(cc->EvalInnerProduct(cY, cW[layer][neurona], input_size), cB[layer][neurona]);

            cNeuronak[neurona] = cc->EvalChebyshevFunction(sigmoid_chebyshev, cNeuronak[neurona], -10, 10, 5);
        }
        cY = cc->EvalMerge(cNeuronak);

        if(depth - cY->GetLevel() < 4){
            //std::cout << "Bootstrapping" << std::endl;
            cY = cc->EvalBootstrap(cY);
        }
    }

    return cY; // prediction
}

KeyPair<DCRTPoly> init_crypto_context_ckks(CryptoContext<DCRTPoly> &cc, nn_t *nn, uint32_t &numSlots){

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
    std::cout << "\nMultDepth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);

    numSlots = 8;
    parameters.SetBatchSize(numSlots);

    // RE-ENCRYPTION
    parameters.SetPREMode(INDCPA);
    parameters.SetKeySwitchTechnique(KeySwitchTechnique::HYBRID);
    parameters.SetRingDim(16384);
    // RE-ENCRYPTION

    cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    cc->Enable(PRE); // PROXY RE-ENCRYPTION

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeysGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

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

    return keys;
}

extern "C" int encrypted_inputs_testing(nn_t *nn, ds_t *ds) {

    // proba
    uint32_t numSlots = 8; // sarearen dimentsioen arabera

    // CryptoContext-a hasieratu
    std::cout << "\nCryptoContext-a sortzen... " << std::endl;
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys_zerb = init_crypto_context_ckks(cc, nn, numSlots);
    if (!keys_zerb.good()) {
        std::cout << "Arazoa zerbitzariaren gakoak sortzean" << std::endl;
        return(-1);
    }

    // RE-ENCRYPTION
    // Bezeroa
    std::cout << "\nB: gako parea sortzen..." << std::endl;
    KeyPair<DCRTPoly> keys_bez = cc->KeyGen(); // Z-ri bidali
    if (!keys_bez.good()) {
        std::cout << "Arazoa bezeroaren gakoak sortzean" << std::endl;
        return(-1);
    }

    std::cout << "\nB --> Z: pk_bez" << std::endl;
    // Zerbitzaria
    std::cout << "\nZ: Re-Encryption Key-a sortzen..." << std::endl;
    EvalKey<DCRTPoly> REK = cc->ReKeyGen(keys_zerb.secretKey, keys_bez.publicKey); // B-ri bidali
    
    std::cout << "\nZ --> B: [REK, pk_zerb(model)]" << std::endl;

    std::cout << "\n\nnumSlots = " << numSlots << std::endl;
    
    int N = ds->n_samples; // input kop
    int input_size = nn->layers_size[0]; // bitarra --> input_size = 2
    std::cout << "N = " << N << ", input_size = " << input_size << std::endl;

    std::vector<std::vector<double>> x(N, std::vector<double>(numSlots, 0.0));
    for(int i=0; i<N; i++){
        for(int j=0; j<input_size; j++){
            x[i][j] = ds->inputs[i*ds->n_inputs + j];
        }
    }

    // Make packed plaintext
    std::vector<Plaintext> px(N);
    for(int i = 0; i < N; i++){
        px[i] = cc->MakeCKKSPackedPlaintext(x[i]);
    }

    std::vector<std::vector<Plaintext>> pW(nn->n_layers - 1);
    std::vector<std::vector<Plaintext>> pB(nn->n_layers - 1);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> cW(N), cB(N);
    //std::vector<Plaintext> pB(nn->n_layers - 1);

    // funtzio bat sortu: pack plaintext
    for (int i = 0; i < nn->n_layers - 1; i++) {
        int rows = nn->layers_size[i];
        int cols = nn->layers_size[i + 1];
        double* Wi = nn->WH[i];

        pW[i].resize(cols);
        cW[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            std::vector<double> Wj(numSlots, 0.0);
            for (int k = 0; k < rows; k++) {
                Wj[k] = Wi[j * rows + k];
            }
            //pW[i][j] = cc->MakeCKKSPackedPlaintext(Wj, 1, depth - 1, nullptr, numSlots);

            pW[i][j] = cc->MakeCKKSPackedPlaintext(Wj);
            cW[i][j] = cc->Encrypt(keys_zerb.publicKey, pW[i][j]);
        }

        std::vector<double> Bi(nn->BH[i], nn->BH[i] + cols);
        pB[i].resize(cols);
        cB[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            std::vector<double> bj(numSlots, 0.0);
            bj[0] = Bi[j];
            //pB[i][j] = cc->MakeCKKSPackedPlaintext(bj, 1, depth - 1, nullptr, numSlots);
            pB[i][j] = cc->MakeCKKSPackedPlaintext(bj);
            cB[i][j] = cc->Encrypt(keys_zerb.publicKey, pB[i][j]);
        }
    }
    
    std::vector<Ciphertext<DCRTPoly>> cY(N); // predictions (y)

    // Denbora
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();

    // paralelizatu --> dummy bat sortu lehenengo
    for(int i = 0; i < N; i++) {
        cY[i] = forward_pass(cc, nn, cW, cB, px[i], input_size);
        std::cout << i << std::endl;
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << "\nDenbora totala: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "BB denbora: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())/N << " ms" << std::endl;


    std::cout << "\nB: REK_cY = REK(pk_zerb(y)) bektorea kalkulatzen..." << std::endl;
    std::vector<Ciphertext<DCRTPoly>> REK_cY(N);
    for(int i = 0; i < N; i++) {
        REK_cY[i] = cc->ReEncrypt(cY[i], REK); // B-ri bidali
    }


    int tp = 0, fp = 0, tn = 0, fn = 0;

    std::cout << "\nB: sk_bez(REK(pk_zerb(y))) = y iragarpen-bektorea deszifratzen...\n" << std::endl;
    std::cout << "\nPredictions = \n";
    for(int i = 0; i < N; i++) {
      Plaintext emaitza;
      cc->Decrypt(keys_bez.secretKey, REK_cY[i], &emaitza); // sk_bez(REK(pk_zerb(y))) = y
      emaitza->SetLength(1);
      double y = emaitza->GetCKKSPackedValue()[0].real();

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
