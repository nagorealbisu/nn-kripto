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

uint32_t depth;
uint32_t numSlots;

void bootstrap(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly> &c, int threshold){
    //std::cout << "c->GetLevel()" << c->GetLevel() << std::endl;
    if(depth - c->GetLevel() < threshold && c->GetLevel() > 0) c = cc->EvalBootstrap(c);
}

void print_net(CryptoContext<DCRTPoly>& cc,
    KeyPair<DCRTPoly>& keys, 
    nn_t* nn, 
    const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& cW,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& cB){

    std::cout.precision(8);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Network:" << std::endl;
    for(int layer=0; layer<nn->n_layers-1; layer++){
        std::cout << "Layer " << layer << " Weights:" << std::endl;
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            std::cout << "Neuron " << neuron << ": [";
            for(int weight=0; weight<nn->layers_size[layer]; weight++){
                Plaintext pW;
                cc->Decrypt(keys.secretKey, cW[layer][neuron][weight], &pW);
                pW->SetLength(numSlots);
                std::cout << pW->GetCKKSPackedValue()[0].real() << " ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Layer " << layer << " Biases:" << std::endl;
        std::cout << "[";
        for(int neuron=0; neuron<nn->layers_size[layer+1]; neuron++){
            Plaintext pB;
            cc->Decrypt(keys.secretKey, cB[layer][neuron], &pB);
            pB->SetLength(numSlots);
            std::cout << pB->GetCKKSPackedValue()[0].real() << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
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
            pB->SetLength(numSlots);
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
                pB->SetLength(numSlots);
                std::cout << pB->GetCKKSPackedValue()[0].real() << " ";
            }
        }
    }
}
