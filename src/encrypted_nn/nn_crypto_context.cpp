/*
  Simple examples for CKKS
 */

#define PROFILE

#include "openfhe.h"
#include <random>
#include <cmath>
#include <iostream>

#include "globals.hpp"

extern "C" {
    #include "nn.h"
    #include "train.h"
}

using namespace lbcrypto;

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

    parameters.SetBatchSize(numSlots); // neurona kopuruarena arabera --> XOR: 2

    std::vector<uint32_t> levelBudget = {3, 3}; // edo {2,2}. {1,1} errorea bootstrapping egiterakoan
    std::vector<uint32_t> bsgsDim = {1, 1};

    uint32_t levelsAfterBootstrap = 20; // 8 ondo
    depth = levelsAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    //std::cout << "\nDepth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);

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

    std::cout << "\nMultDepth: " << depth << std::endl;

    return keys;
}
