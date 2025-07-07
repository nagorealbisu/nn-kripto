#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ds.h"
#include "nn.h"
#include "utils.h"
#include "globals.h"
#include "test.h"
#include "train.h"

#include "enc_testing.h"
#include "enc_training.h"

int main(int argc, char **argv) {

    ds_t ds;
    nn_t nn;
    
    if(argc == 1){
        printf("No arguments passed!\n");
        exit(0);
    }

    parse_arguments(argc, argv);

    if(train_mode){
        //srand(seed);
        read_csv(dataset, &ds, layers[0], layers[n_layers - 1]);
        init_nn(&nn, n_layers, layers);

        printf("\nPLAINTEXT TRAINING\n\n");
        import_nn(&nn, model);
        print_nn(&nn);
        train(&nn, &ds, epochs, batches, lr);
        print_nn(&nn);
        
        printf("\nCIPHERTEXT TRAINING\n\n");
        import_nn(&nn, model);
        print_nn(&nn);
        //encrypt_dataset(); // Serialization egiten ikasi
        encrypted_dataset_training(&nn, &ds, epochs, batches, lr);
        print_nn(&nn);
        export_nn(&nn, model);
    }
    else if(test_mode){
        import_nn(&nn, model);
        read_csv(dataset, &ds, nn.layers_size[0], nn.layers_size[n_layers - 1]);
        printf("\nPLAINTEXT INPUT INFERENCE\n\n");
        test(&nn, &ds);
	    print_nn(&nn);
        printf("\nCIPHERTEXT INPUT INFERENCE\n");
	    encrypted_dataset_testing(&nn, &ds);
    }
    
    return(0);
}

