#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ds.h"
#include "nn.h"
#include "nn_aux.h"
#include "utils.h"
#include "matrix.h"
#include "test.h"
#include "train.h"
#include "globals.h"

#include <unistd.h>

void init_nn(nn_t *nn, int n_layers, int *layers_size){
    nn->n_layers = n_layers;
    nn->layers_size = layers_size;
    nn->init_weight_ptr = init_weight_rnd;
    
    parse_activation(nn, activation);
    parse_loss(nn, loss); 
    nn->BH = alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
    
}

#ifdef CPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
  
    order = (int*)malloc(ds->n_samples * sizeof(int));
    for(i=0; i<nn->n_layers; i++){
        printf("nn->layers_size: %d\n", nn->layers_size[i]);
    }
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n = 0; n < epochs; n++) {
        
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;

        // shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);

        for (x = 0; x < n_batches; x++) {
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
            
                i = order[min_batch];
                //printf("\nFORWARD PASS");
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                //printf("\nBACK PROPAGATION");
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
            }
            //printf("\nUPDATE");
            update(nn, D, d, lr, size_batch);
            //print_nn(nn);
        }
        loss /= ds->n_samples; 
        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss);

        //sleep(1);
        print_nn(nn);
    }
}


void train2(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
  
    order = (int*)malloc(ds->n_samples * sizeof(int));
    for(i=0; i<nn->n_layers; i++){
        printf("nn->layers_size: %d\n", nn->layers_size[i]);
    }
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs;n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);

        for (x = 0; x < n_batches; x++) {
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
            
                i = order[min_batch];
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
            }
            
            update(nn, D, d, lr, size_batch);
        }
        loss /= ds->n_samples; 
        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss);

        //sleep(1);
        //print_nn(nn);
    }

}

void test(nn_t *nn, ds_t *ds){
    
    int i;
    double **A;
    int predicted = 0; 
    int expected = 0;
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    
    for(i = 0; i < ds->n_samples; i++){

        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
      
        expected = ds->outputs[i];
        predicted = (A[nn->n_layers - 1][0] >= 0.5) ? 1 : 0;
        if (expected==1)
        {
          printf("A[nn->n_layers - 1][0] cuando expected class = 1 : %f\n", A[nn->n_layers - 1][0]);
          if (predicted==1)
          {
            tp++;
          }
          else
          {
            fn++;
          }
        }
        else //expected 0
        {
          printf("A[nn->n_layers - 1][0] cuando expected class = 0 : %f\n", A[nn->n_layers - 1][0]);
          if (predicted==1)
          {
            fp++;
          }
          else
          {
            tn++;
          }
        }
    }
    printf("TP = %d, FP = %d\n", tp, fp);
    printf("FN = %d, TN = %d\n", fn, tn);

    double p = precision(tp, fp);
    double r = recall(tp, fn);
    printf("\nPrecision = %f, \n", p);
    printf("Recall = %f, \n", r);
    printf("F1 = %f, \n", f1(p, r));
  
    matrix_free(*A);
}

#endif

void print_nn(nn_t *nn){

    int i, j, k;
    
    printf("Layers (I/H/O)\n");

    for (i = 0; i < nn->n_layers; i++) {
        printf("%d ", nn->layers_size[i]);
    }
    printf("\n");
    
    printf("Hidden Biases\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            printf("%lf ", nn->BH[i][j]);
        }
        printf("\n");
    }

    printf("Hidden Weights\n ");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                printf("%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            printf("\n");
        }
    }

}

void import_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"r")) == NULL){
        perror("Error importing the model\n");
        exit(1);
    }
    
    fscanf(fd, "%d ", &n_layers);

    layers = (int*)malloc(n_layers * sizeof(int));

    for (i = 0; i < n_layers; i++) {
        fscanf(fd, "%d ", &(layers[i]));
    }

    init_nn(nn, n_layers, layers);
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(nn->BH[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(nn->WH[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }
    fclose(fd);
}

void export_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"w")) == NULL){
        perror("Error exporting the model");
        exit(1);
    }
    
    fprintf(fd, "%d\n", nn->n_layers);

    for (i = 0; i < nn->n_layers; i++) {
        fprintf(fd, "%d ", nn->layers_size[i]);
    }
    fprintf(fd, "\n");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", nn->BH[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }
    fclose(fd);
}

