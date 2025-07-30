# MPL for Encrypted Training and Testing

This repository implements a Multilayer Perceptron (MLP) designed for encrypted training and testing scenarios, suitable for research in privacy-preserving machine learning.

## Execution

make

### XOR TRAIN
./bin/train --train --learning_rate 0.1 --epochs 10000 --batch_number 1 --dataset datasets/xor.csv --layers 2,2,1 -s 1 --model models/model_start_xor.m --verbose --activation sigmoid --loss mse

### XOR TEST
./bin/test --test  --dataset datasets/xor.csv --model models/model_xor.m --verbose --activation sigmoid 

### Diabetes TRAIN
./bin/train --train --learning_rate 0.01 --epochs 2500 --batch_number 1 --dataset datasets/diabetes_train.csv --layers 8,3,1 -s 1 --model model.m --verbose --scaling normalization

### Diabetes TEST
./bin/test --test  --dataset datasets/diabetes_test.csv --model model.m --verbose --activation sigmoid --scaling normalization


## CHEBYSHEV C liburutegia: https://people.sc.fsu.edu/~jburkardt/c_src/chebyshev/chebyshev.html

gcc -Iinclude -o chebyshev_test chebyshev_test.c src/chebyshev.c -lm
./chebyshev_test
