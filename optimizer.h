#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

#include "network.h"

typedef struct {
  int batchsize;
  int epoch;
  double lambda;
  double learning_rate;
} OptimizerParameters;

int fit(Network* p, double** x_train, double** y_train, int x_rows, int x_cols, int y_rows, int y_cols, const OptimizerParameters* params);

#endif
