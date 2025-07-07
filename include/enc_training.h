#ifndef ENC_TRAIN_HPP
#define ENC_TRAIN_HPP

#ifdef __cplusplus
extern "C" {
#endif

int encrypted_dataset_training(nn_t *nn, ds_t *ds, int epochs, int batches, double lr);

#ifdef __cplusplus
}
#endif

#endif // ENC_TRAIN_HPP
