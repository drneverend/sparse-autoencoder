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
