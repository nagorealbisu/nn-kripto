#include "nn.h"
#include "matrix.h"
#include "test.h"
#include "chebyshev.h"
    
void forward_pass_test(nn_t *nn, double *input, double **A){

    int i;
    //printf("chebyshev\n");
    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    //double max = 0;
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
        /*double bal_max = A[i][0];
        for(int j = 0; j < nn->layers_size[i]; j++) {
            if(A[i][j] > bal_max) {
                bal_max = A[i][j];
            }
        }*/
        // ALDATUTA
        double *c = chebyshev_coefficients(-10, 10, 10, nn->activation_ptr[i - 1]);
        matrix_func_chebyshev(A[i], A[i], nn->layers_size[i], 1, -10, 10, 10, c);
        // ALDATUTA
        //max = bal_max;
    }
    //printf("Balio maximoa: %f\n", max);
}

void forward_pass_test2(nn_t *nn, double *input, double **A){

    int i;
    //printf("normal\n");
    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }
}

//printf("Expected: %f , Obtained: %f Loss %f\n", output[0], A[nn->n_layers - 1][0], loss);


float precision(int tp, int fp){
    if (tp + fp == 0) /* avoid division by zero */
        return 0.0f;
    return (float) tp / (tp + fp);
}

float recall(int tp, int fn){
    if (tp + fn == 0) /* avoid division by zero */
        return 0.0f;
    return (float) tp / (tp + fn);
}

float f1(float p, float r){
    if (p + r == 0) /* avoid division by zero */
      return 0.0f;
    return 2*p*r/(p+r);
}
