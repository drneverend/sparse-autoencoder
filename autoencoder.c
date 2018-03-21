#include <stdio.h>

#include "layer.h"
#include "network.h"
#include "optimizer.h"
#include "dataset.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("autoencoder <data file>\n");
    return 0;
  }

  const char* datafile = argv[1];

  int samples = 0, width = 0, height = 0;
  double*** data = NULL;
  get_dataset_size(datafile, &samples, &width, &height);
  read_dataset(datafile, &data);

  Network* network = create_network();
  add_layer(network, 10, ActivateSigmoid);
  add_layer(network, 5, ActivateSigmoid);
  add_layer(network, 10, ActivateLinear);

  fit(network, data, data);

  destroy_network(network);
  return 0;
}
