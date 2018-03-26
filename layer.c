#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "layer.h"
#include "linearalgebra.h"
#include "util.h"

#define SAFE_DELETE(p) if (p != NULL) { free(p); }

Layer* create_layer(int nodes,int input_nodes, ActivateType activate) {
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  layer->nodes = nodes;
  layer->input_nodes = input_nodes;

  layer->weights = create_2d_array(nodes, input_nodes);
  layer->gradient = create_2d_array(nodes, input_nodes);
  layer->batch_gradient = create_2d_array(nodes, input_nodes);

  layer->bias = (double*)malloc(nodes * sizeof(double));
  layer->gradient_bias = NULL; /* just keep a null pointer. it will be set during optimization */
  layer->batch_gradient_bias = (double*)malloc(nodes * sizeof(double));

  layer->z = (double*)malloc(nodes * sizeof(double));
  layer->a = (double*)malloc(nodes * sizeof(double));
  layer->delta = (double*)malloc(nodes * sizeof(double));

  zero_matrix_2(layer->batch_gradient, nodes, input_nodes);
  zero_matrix_1(layer->batch_gradient_bias, nodes);

  return layer;
}
void destroy_layer(Layer* layer) {
  if (layer != NULL) {
    destroy_2d_array(layer->weights, layer->nodes);
    destroy_2d_array(layer->gradient, layer->nodes);
    destroy_2d_array(layer->batch_gradient, layer->nodes);
    SAFE_DELETE(layer->bias)
    SAFE_DELETE(layer->z)
    SAFE_DELETE(layer->a)
    SAFE_DELETE(layer->delta)
    SAFE_DELETE(layer->batch_gradient_bias)
    free(layer);
  }
}

double sigmoid(double v) {
  return 1.0 / (1 + exp(-v));
}

void sigmoid_a(double* a, double* result, int n) {
  for (int i = 0; i < n; i++) {
    result[i] = sigmoid(a[i]);
  } 
}

void forward_layer(Layer* layer, double* input) {
  matrix_multi_21(layer->weights, layer->nodes, layer->input_nodes, input, layer->z);
  matrix_add_1(layer->z, layer->bias, layer->z, layer->nodes);
  sigmoid_a(layer->z, layer->a, layer->nodes);
}

void delta_last(double* y, double* a, double* result, int n) {
  for (int i = 0; i < n; i++) {
    result[i] = -(y[i] - a[i]) * a[i] * (1 - a[i]);
  }
}

void delta_mid(double** weight, int rows, int cols, double* delta, double* a, double* result) {
  for (int i = 0; i < cols; i++) {
    double s = 0;
    for (int j = 0; j < rows; j++) {
      s += weight[j][i] * delta[j];
    }
    result[i] = s * a[i] * (1 - a[i]);
  }
}

void backward_layer(Layer* layer, Layer* next, double* y) {
  if (next == NULL) { // it is the last layer
    delta_last(y, layer->a, layer->delta, layer->nodes);
  } else {
    delta_mid(next->weights, next->nodes, next->input_nodes, next->delta, layer->a, layer->delta);
  }
}

void gradient(Layer* layer, Layer* last, double* x) {
  if (last == NULL) {
    cross(layer->delta, layer->nodes, x, layer->input_nodes, layer->gradient);
  } else {
    cross(layer->delta, layer->nodes, last->a, layer->input_nodes, layer->gradient);
  }
  layer->gradient_bias = layer->delta;
  matrix_add_2(layer->gradient, layer->batch_gradient, layer->batch_gradient, layer->nodes, layer->input_nodes);
  matrix_add_1(layer->gradient_bias, layer->batch_gradient_bias, layer->batch_gradient_bias, layer->nodes);
}

void update_weights(Layer* layer, double lambda, double lr, int batch_size) {
  /*update weights and bias, and reset batch gradients*/
  for (int i = 0; i < layer->nodes; i++) {
    for (int j = 0; j < layer->input_nodes; j++) {
      layer->weights[i][j] = layer->weights[i][j] - lr * (layer->batch_gradient[i][j] / batch_size + lambda * layer->weights[i][j]);
    }
  }
  for (int i = 0; i < layer->nodes; i++) {
    layer->bias[i] = layer->bias[i] - lr * (layer->batch_gradient_bias[i] / batch_size);
  }

  zero_matrix_2(layer->batch_gradient, layer->nodes, layer->input_nodes);
  zero_matrix_1(layer->batch_gradient_bias, layer->nodes);
}
