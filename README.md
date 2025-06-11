# TRAINING AND TESTING XOR

## XOR TRAIN
./bin/nn.exe --train --learning_rate 0.1 --epochs 10000 --batch_number 1 --dataset datasets/xor.csv --layers 2,2,1 -s 1 --model models/model_xor.m --verbose --activation sigmoid --loss mse

## XOR TEST
./bin/nn --test  --dataset datasets/xor.csv --model models/model_xor.m --verbose --activation sigmoid 

## Diabetes TRAIN
./bin/nn --train --learning_rate 0.01 --epochs 2500 --batch_number 1 --dataset datasets/diabetes_train.csv --layers 8,3,1 -s 1 --model model.m --verbose --scaling normalization

## Diabetes TEST
./bin/nn --test  --dataset datasets/diabetes_test.csv --model model.m --verbose --activation sigmoid --scaling normalization


# CHEBYSHEV C liburutegia: https://people.sc.fsu.edu/~jburkardt/c_src/chebyshev/chebyshev.html

gcc -Iinclude -o chebyshev_test chebyshev_test.c src/chebyshev.c -lm
./chebyshev_test


## OUTPUT XOR TESTING
$ ./bin/nn --test  --dataset datasets/xor.csv --model models/model_xor.m --verbose --activation sigmoid

PLAINTEXT INPUT INFERENCE

A[nn->n_layers - 1][0] cuando expected class = 0 : 0.001267
A[nn->n_layers - 1][0] cuando expected class = 1 : 0.998550
A[nn->n_layers - 1][0] cuando expected class = 1 : 0.998547
A[nn->n_layers - 1][0] cuando expected class = 0 : 0.001605
TP = 2, FP = 0
FN = 0, TN = 2

Precision = 1.000000,
Recall = 1.000000,
F1 = 1.000000,
Layers (I/H/O)
2 2 1
Hidden Biases
 -2.680850 -6.433990
-3.905363
Hidden Weights
 6.178459 6.172641
4.194540 4.193236
8.544668 -9.231778

CIPHERTEXT INPUT INFERENCE
Gakoak sortzen...
N = 4, input_size = 2
0
1
2
3

Denbora Encrypted Testing: 3779 ms

Predictions =
Prediction (real): 0.0192614, Prediction: 0, Output: 0
Prediction (real): 1.04428, Prediction: 1, Output: 1
Prediction (real): 1.04384, Prediction: 1, Output: 1
Prediction (real): 0.00422179, Prediction: 0, Output: 0
TP: 2, FP: 0
FN: 0, TN: 2
Precision: 1
Recall: 1
F1: 1
