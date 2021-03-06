#ifndef __NETWORK_H
#define __NETWORK_H

#include "layer.h"

typedef struct struct_LayerNode {
  struct struct_LayerNode* prev;
  struct struct_LayerNode* next;
  Layer* layer;
} LayerNode;

typedef struct {
  LayerNode* head;
  LayerNode* tail;
  int layers;
  int input_dim;
} Network;

typedef struct {
  LayerNode* node;
  int forward;
} NetworkIterator;

Network* create_network();
void destroy_network(Network* network);
Network* add_layer(Network* network, int nodes, ActivateType activate);

void set_forward_iterator(Network* network, NetworkIterator* iter);
void set_backward_iterator(Network* network, NetworkIterator* iter);
void next_layer(NetworkIterator* iter);
Layer* get_layer(NetworkIterator* iter);
int has_layer(NetworkIterator* iter);

Layer* output_layer(Network* network);

double mse_loss(Network* network, double* y);
double regular_loss(Network* network);
void write_weights(Network* network);

#endif
