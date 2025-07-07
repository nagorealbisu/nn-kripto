#ifndef ENC_TRAIN_FUNCTIONS_HPP
#define ENC_TRAIN_FUNCTIONS_HPP

#include "openfhe.h"
#include "nn.h"
#include "train.h"

using namespace lbcrypto;

inline double sigmoid_chebyshev(double x) {
    return 1.0 / (1.0 + std::exp(-x));
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
);

Ciphertext<DCRTPoly> mse(
    CryptoContext<DCRTPoly>& cc,
    std::vector<Ciphertext<DCRTPoly>>& cA,
    std::vector<Ciphertext<DCRTPoly>>& cY,
    int length
);

Ciphertext<DCRTPoly> dmse(
    CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& cA,
    Ciphertext<DCRTPoly>& cY,
    int length
);

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
    std::vector<Ciphertext<DCRTPoly>>& cy
);

void update(
    CryptoContext<DCRTPoly> &cc,
    const nn_t* nn,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB,
    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cD,
    std::vector<std::vector<Ciphertext<DCRTPoly>>>& cd,
    double lr,
    int batch_size
);

#endif // ENC_TRAIN_FUNCTIONS_HPP
