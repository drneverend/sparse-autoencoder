#ifndef __NETWORK_H
#define __NETWORK_H

#include "layer.h"

typedef struct {
  Layer* prev;
  Layer* next;
  Layer* node;
} LayerNode;

typedef struct {
  LayerNode* head;
  LayerNode* last;
  
} Network;

typedef struct {
  LayerNode* start;
  LayerNode* end;
  int forward;
} NetworkIterator;


Network* create_network();
void destroy_network(Network* p);
Network* add_layer(Network* p, int nodes, ActivateType activate);

void set_forward_iterator(Network*p, NetworkIterator* iter);
void set_backward_iterator(Network*p, NetworkIterator* iter);
void next_layer(NetworkIterator* iter);
void no_layer(NetworkIterator* iter);

#endif
