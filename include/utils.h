#ifndef _UTILS_H
#define _UTILS_H

#include "ds.h"
#include "globals.h"
#include "nn.h"
#include <time.h>

char *strremove(char *str, const char *sub);

void parse_arguments(int argc, char **argv);

void parse_loss(nn_t *nn, char *optarg);

void parse_scaling(ds_t *ds, char *optarg);

void parse_activation(nn_t *nn, char *optarg);

long diff_time(const struct timespec t2, const struct timespec t1);

#endif
