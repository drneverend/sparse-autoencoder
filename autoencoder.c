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
  double** data = NULL;
  get_dataset_size(datafile, &samples, &width, &height);
  read_dataset(datafile, &data);

  Network* network = create_network(64);
  add_layer(network, 25, ActivateSigmoid);
  add_layer(network, 64, ActivateLinear);

  OptimizerParameters params;
  params.batchsize = 64;
  params.epoch = 10;
  params.lambda = 0.0001;
  params.learning_rate = 0.1;
  fit(network, data, data, samples, width * height, samples, width * height, &params);

  destroy_network(network);
  return 0;
}
