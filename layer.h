#ifndef __LAYER_H
#define __LAYER_H

#include "activates.h"

typedef struct {
  int nodes;
  int input_nodes;
  double** weights;
  double* bias;
  double* z;
  double* a;
  double* delta;
  double** gradient;
  double** batch_gradient;
  double* gradient_bias;
  double* batch_gradient_bias;
} Layer;

Layer* create_layer(int nodes,int input_nodes, ActivateType activate);
void destroy_layer(Layer* layer);

void forward_layer(Layer* layer, double* input);
void backward_layer(Layer* layer, Layer* next, double* y);
void gradient(Layer* layer, Layer* last, double* x);

void update_weights(Layer* layer, double lambda, double lr, int batch_size);

#endif
