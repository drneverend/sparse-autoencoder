#include <stdlib.h>
#include <stdio.h>

#include "errors.h"
#include "optimizer.h"
#include "network.h"

void swap(int* d, int a, int b) {
  int t = d[a];
  d[a] = d[b];
  d[b] = t;
}

int* gen_index(int n, int batchsize, int* index_size) {
  int pad_size = (n + batchsize - 1) / batchsize * batchsize;
  *index_size = pad_size;

  int* indices = (int*)malloc(pad_size * sizeof(int));

  int i;
  for (i = 0; i < n; i++) {
    indices[i] = i; 
  }
  for (i = n; i < pad_size; i++) {
    indices[i] = i - (pad_size - n);
  }

  for (i = 0; i < 2 * n; i++) {
    int src = rand() * pad_size / RAND_MAX;
    int dst = rand() * pad_size / RAND_MAX;
    swap(indices, src, dst);
  }

  return indices;
}

int fit(Network* network, double** x_train, double** y_train, int x_rows, int x_cols, int y_rows, int y_cols, const OptimizerParameters* params) {
  srand(12345);

  if (x_rows != y_rows || x_cols != y_cols) {
    return ERROR_INVALID_PARAM;
  }

  int index_size = 0;
  int* indices = gen_index(x_rows, params->batchsize, &index_size);
  int nbatch = index_size / params->batchsize;

  for (int i = 0; i < params->epoch; i++) {
    printf("epoch %d\n", i);
    for (int j = 0; j < nbatch; j++) {
      double loss = 0;
      for (int m = 0; m < params->batchsize; m++) {
        int data_index = indices[j * params->batchsize + m];
        double* data = x_train[data_index];
        double* y_data = y_train[data_index];

        /*forward pass*/
        NetworkIterator iter;
        set_forward_iterator(network, &iter);
        Layer* lastlayer = NULL;
        while (has_layer(&iter)) {
          Layer* layer = get_layer(&iter);
          if (lastlayer == NULL) {
            forward_layer(layer, data);
          } else {
            forward_layer(layer, lastlayer->a);
          }
          lastlayer = layer;
          next_layer(&iter);
        }

        /*compute mse loss*/
        loss += mse_loss(network, y_data);

        /*compute delta*/
        set_backward_iterator(network, &iter);
        lastlayer = NULL;
        while (has_layer(&iter)) {
          Layer* layer = get_layer(&iter);
          backward_layer(layer, lastlayer, data);
          lastlayer = layer;
          next_layer(&iter);
        }

        /*compute weights and update weights at the last sample in the batch*/
        set_forward_iterator(network, &iter);
        lastlayer = NULL;
        while (has_layer(&iter)) {
          Layer* layer = get_layer(&iter);
          gradient(layer, lastlayer, data);
          lastlayer = layer;
          next_layer(&iter);
        }
        if (m >= params->batchsize - 1) {
          set_forward_iterator(network, &iter);
          while (has_layer(&iter)) {
            Layer* layer = get_layer(&iter);
            update_weights(layer, params->lambda, params->learning_rate, params->batchsize);
            next_layer(&iter);
          }
        }
      }
      loss /= params->batchsize;
      loss += params->lambda * regular_loss(network);
      printf("loss: %f\r", loss);
    }
    printf("\n");
  }

  free(indices);

  return SUCCESS_OK;
}

