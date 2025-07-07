//#ifdef CPU

#include "train.h"
#include "chebyshev.h"
#include <unistd.h>

void forward_pass(nn_t *nn, double *input, double **A, double **Z){

    int i;

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    for(i = 1; i < nn->n_layers; i++){

        //printf("\nwx + b...\n");
        matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  

        // ALDATUTA
        double *c_act = chebyshev_coefficients(-10, 10, 10, nn->activation_ptr[i - 1]);
        //double *c_dact = chebyshev_coefficients(-10, 10, 10, nn->dactivation_ptr[i - 1]);
        //printf("\nSigmoid...\n");
        matrix_func_chebyshev(A[i], Z[i], nn->layers_size[i], 1, -10, 10, 10, c_act);
        //printf("\ndSigmoid...\n");
        //matrix_func_chebyshev(Z[i], Z[i], nn->layers_size[i], 1, -10, 10, 10, c_dact);
        for (int j = 0; j < nn->layers_size[i]; j++) {
            // dSigmoid = A[i][j] * (1 - A[i][j])
            Z[i][j] = A[i][j] * (1.0 - A[i][j]);
        }

    }
}

void forward_pass2(nn_t *nn, double *input, double **A, double **Z){

    int i;

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    //double max = 0;
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        /*
        double bal_max = Z[i][0];
        for(int j = 0; j < nn->layers_size[i]; j++) {
            if(Z[i][j] > bal_max) {
                bal_max = Z[i][j];
            }
        }*/
        matrix_func(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
        matrix_func(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
        //max = bal_max;
    }
    //printf("Balio maximoa: %f\n", max);
}

double back_prop(nn_t *nn, double *output, double **A, double **Z, double **D, double **d){

    int i, n_l;
    int *l_s;
    double loss;
    double *T;
    double **E, **D_aux;

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    D_aux = alloc_matrix_2v(n_l - 1, &(l_s[1]), &(l_s[0]), init_zero);
    E = alloc_matrix_1v(n_l - 1, &(l_s[1]), init_zero);

    loss = nn->loss(A[n_l - 1], output, l_s[n_l - 1]);

    //matrix_sub(E[n_l - 2], A[n_l - 1], output, l_s[n_l - 1], 1);
    for (i=0; i<l_s[n_l-1]; i++)
    {
      *m_elem(E[n_l-2],1,i,0) = nn->dloss(A[n_l-1],output,1);
    }

    matrix_mul_dot(E[n_l - 2], E[n_l - 2], Z[n_l - 1], l_s[n_l - 1], 1);  

    T = matrix_transpose(A[n_l - 2], l_s[n_l - 2], 1);
    matrix_mul(D_aux[n_l - 2], E[n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    matrix_free(T);

    matrix_sum(D[n_l - 2], D[n_l - 2], D_aux[n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(d[n_l - 2], d[n_l - 2], E[n_l - 2], l_s[n_l - 1], 1);

    //printf("\nHidden layers:");
    for (i = n_l - 2; i > 0; i--) {
            
        T = matrix_transpose(nn->WH[i], l_s[i + 1], l_s[i]);
        matrix_mul(E[i - 1], T, E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        matrix_free(T);

        matrix_mul_dot(E[i - 1], E[i - 1], Z[i], l_s[i], 1);

        matrix_mul(D_aux[i - 1], E[i - 1], A[i - 1], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(D[i - 1], D[i - 1], D_aux[i - 1], l_s[i], l_s[i - 1]);
        matrix_sum(d[i - 1], d[i - 1], E[i - 1], l_s[i], 1);
    }

    matrix_free_2D(D_aux, n_l - 1);
    matrix_free_2D(E, n_l - 1);

    return(loss);

}

double back_prop2(nn_t *nn, double *output, double **A, double **Z, double **D, double **d){

    int i, n_l;
    int *l_s;
    double loss;
    double *T;
    double **E, **D_aux;

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    D_aux = alloc_matrix_2v(n_l - 1, &(l_s[1]), &(l_s[0]), init_zero);
    E = alloc_matrix_1v(n_l - 1, &(l_s[1]), init_zero);

    loss = nn->loss(A[n_l - 1], output, l_s[n_l - 1]);

    //matrix_sub(E[n_l - 2], A[n_l - 1], output, l_s[n_l - 1], 1);
    for (i=0; i<l_s[n_l-1]; i++)
    {
      *m_elem(E[n_l-2],1,i,0) = nn->dloss(A[n_l-1],output,1);
    }
    matrix_mul_dot(E[n_l - 2], E[n_l - 2], Z[n_l - 1], l_s[n_l - 1], 1);  
    

    T = matrix_transpose(A[n_l - 2], l_s[n_l - 2], 1);
    matrix_mul(D_aux[n_l - 2], E[n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    matrix_free(T);

    matrix_sum(D[n_l - 2], D[n_l - 2], D_aux[n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(d[n_l - 2], d[n_l - 2], E[n_l - 2], l_s[n_l - 1], 1);

    for (i = n_l - 2; i > 0; i--) {
            
        T = matrix_transpose(nn->WH[i], l_s[i + 1], l_s[i]);
        matrix_mul(E[i - 1], T, E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        matrix_free(T);

        matrix_mul_dot(E[i - 1], E[i - 1], Z[i], l_s[i], 1);

        matrix_mul(D_aux[i - 1], E[i - 1], A[i - 1], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(D[i - 1], D[i - 1], D_aux[i - 1], l_s[i], l_s[i - 1]);
        matrix_sum(d[i - 1], d[i - 1], E[i - 1], l_s[i], 1);
    }

    matrix_free_2D(D_aux, n_l - 1);
    matrix_free_2D(E, n_l - 1);

    return(loss);

}

void update(nn_t *nn, double **D, double **d, double lr, int batch_size){

    int i;

    for(i = 0; i < nn->n_layers - 1; i++){

        matrix_mul_cnt(D[i], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));

        matrix_mul_cnt(d[i], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));

        matrix_sub(nn->WH[i], nn->WH[i], D[i],  nn->layers_size[i + 1], nn->layers_size[i]);
        matrix_sub(nn->BH[i], nn->BH[i], d[i],  nn->layers_size[i + 1], 1);
        matrix_zero(D[i], nn->layers_size[i + 1], nn->layers_size[i]);
        matrix_zero(d[i], nn->layers_size[i + 1], 1);
    }
}

//#endif
