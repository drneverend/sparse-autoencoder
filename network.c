#include <stdlib.h>
#include <stdio.h>
#include "network.h"

Network* create_network(int input_dim) {
  Network* result = (Network*)malloc(sizeof(Network));
  result->head = NULL;
  result->tail = NULL;
  result->layers = 0;
  result->input_dim = input_dim;

  return result;
}

void destroy_network(Network* network) {
  if (network != NULL) {
    LayerNode* node = network->head;
    while (node != NULL) {
      destroy_layer(node->layer);
      LayerNode* last = node;
      node = node->next;
      free(last);
    }
    network->head = NULL;
    network->tail = NULL;
    network->layers = 0;
    free(network);
  }
}

Network* add_layer(Network* network, int nodes, ActivateType activate) {
  int input_nodes = network->tail == NULL ? network->input_dim : network->tail->layer->nodes;
  Layer* layer = create_layer(nodes, input_nodes, activate);
  LayerNode* node = (LayerNode*)malloc(sizeof(LayerNode));
  node->layer = layer;
  node->next = NULL;
  node->prev = network->tail;
  if (network->head == NULL) {
    network->head = node;
  }
  if (network->tail != NULL) {
    network->tail->next = node;
  }
  network->tail = node;
  return network;
}

void set_forward_iterator(Network*network, NetworkIterator* iter) {
  iter->forward = 1;
  iter->node = network->head;
}
void set_backward_iterator(Network*network, NetworkIterator* iter) {
  iter->forward = 0;
  iter->node = network->tail;
}
void next_layer(NetworkIterator* iter) {
  if (iter != NULL && iter->node != NULL) {
    if (iter->forward == 1) {
      iter->node = iter->node->next;
    } else {
      iter->node = iter->node->prev;
    }
  } else {
    printf("moving the iterator while it reaches end");
  }
}
int has_layer(NetworkIterator* iter) {
  return iter->node != NULL;
}

Layer* get_layer(NetworkIterator* iter) {
  return iter->node->layer;
}

Layer* output_layer(Network* network) {
  if (network != NULL) {
    return network->tail->layer;
  } else {
    return NULL;
  }
}

double mse_loss(Network* network, double* y) {
  Layer* outputlayer = output_layer(network);
  int output_nodes = outputlayer->nodes;
  double* output = outputlayer->a;

  double loss = 0;
  for (int i = 0; i < output_nodes; i++) {
    loss += (y[i] - output[i]) * (y[i] - output[i]);
  }

  return loss / 2;
}

double regular_loss(Network* network) {
  double loss = 0;

  LayerNode* node = network->head;
  while (node != NULL) {
    Layer* layer = node->layer;
    double** weights = layer->weights;
    for (int i = 0; i < layer->nodes; i++) {
      for (int j = 0; j < layer->input_nodes; j++) {
        loss += weights[i][j] * weights[i][j];
      }
    }
    node = node->next; 
  }

  return loss / 2;
}

void write_layer_weights(Layer* layer, const char* filename) {
  FILE* f = fopen(filename, "wb");
  for (int i = 0; i < layer->nodes; i++) {
    fwrite(layer->weights[i], sizeof(double), layer->input_nodes, f);
  }
  fclose(f);
}

void write_weights(Network* network) {
  LayerNode* node = network->head;
  int i = 0;
  char filenames[1024] = {0};
  while (node != NULL) {
    Layer* layer = node->layer;
    sprintf(filenames, "layer_weights_%d.data", i);
    write_layer_weights(layer, filenames);
    node = node->next; 
    i++;
  }
}
