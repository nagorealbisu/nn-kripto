#ifndef NN_CRYPTO_CONTEXT_HPP
#define NN_CRYPTO_CONTEXT_HPP

#include "openfhe.h"
#include <vector>

extern "C" {
    #include "nn.h"
    #include "train.h"
}

using namespace lbcrypto;

extern uint32_t depth;
extern uint32_t numSlots;

void bootstrap(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly> &c, int threshold);

void print_net(CryptoContext<DCRTPoly>& cc,
               KeyPair<DCRTPoly>& keys,
               nn_t* nn,
               const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
               const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB);

void init_zero(std::vector<std::vector<Ciphertext<DCRTPoly>>>& cM,
               const Ciphertext<DCRTPoly>& cZero);

void init_zero_3D(std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cM,
                  const Ciphertext<DCRTPoly>& cZero);

void print_matrix(std::vector<std::vector<Ciphertext<DCRTPoly>>>& cM,
                  KeyPair<DCRTPoly> keys,
                  CryptoContext<DCRTPoly>& cc);

void print_matrix_3D(std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cM,
                     KeyPair<DCRTPoly> keys,
                     CryptoContext<DCRTPoly>& cc);

#endif // NN_CRYPTO_CONTEXT_HPP
