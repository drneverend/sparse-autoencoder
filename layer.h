#ifndef __LAYER_H
#define __LAYER_H

#include "activates.h"

typedef struct {
  int nodes;
  double** weights;
  double* bias;
  double* z;
  double* a;
  double* delta;
  double** gradient;
  double** batch_gradient;
} Layer;

Layer* create_layer(int nodes,int input_nodes, ActivateType activate);
void destroy_layer(Layer* p);

void forward_layer(Layer* p, double* input);
void backward_layer(Layer* p, Layer* next);

void start_batch(Layer* p);
void stop_batch(Layer* p);

#endif
