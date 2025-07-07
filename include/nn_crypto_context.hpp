#ifndef CKKS_UTILS_HPP
#define CKKS_UTILS_HPP

#include "openfhe.h"
#include "nn.h"
#include "train.h"

using namespace lbcrypto;

extern uint32_t depth;

KeyPair<DCRTPoly> init_crypto_context_ckks_train(
    CryptoContext<DCRTPoly> &cc,
    nn_t* nn,
    uint32_t &numSlots
);

#endif // CKKS_UTILS_HPP
