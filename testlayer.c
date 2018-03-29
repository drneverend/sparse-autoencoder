#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "layer.h"

void compare_1(double* exp, double* real, int n, const char* indicator) {
  for (int i = 0; i < n; i++) {
    if (fabs(real[i] - exp[i]) > 0.00001) {
      printf("%s[%d]: %f -> %f\n", indicator, i, exp[i], real[i]);
    }
  }
}

void compare_2(double exp[2][3], double** real, int rows, int cols, const char* indicator) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (fabs(real[i][j] - exp[i][j]) > 0.00001) {
        printf("%s[%d]: %f -> %f\n", indicator, i, exp[i][j],real[i][j]);
      }
    }
  }
}

int main() {
  Layer* layer = create_layer(2,3,ActivateSigmoid);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      layer->weights[i][j] = (i * 3 + j) / 10.0;
    }
    layer->bias[i] = i * 0.5;
  }
  double input[3] = {0.1,0.2,0.3};
  forward_layer(layer, &input[0]);
  double expectedz[2] = {0.08,0.76};
  compare_1(expectedz, layer->z, 2, "z");
  double expecteda[2] = {0.51999,0.68135};
  compare_1(expecteda, layer->a, 2, "a");
  double y[2] = {0.8,0.9};
  backward_layer(layer, NULL, &y[0]);
  double expectedd[2] = {-0.069891,-0.047470};
  compare_1(expectedd, layer->delta, 2, "delta");
  gradient(layer, NULL, input);
  double expectedg[2][3] = { {-0.0069891, -0.0139782, -0.0209672}, {-0.0047470, -0.0094941, -0.0142411}};
  compare_2(expectedg, layer->gradient, 2, 3, "gradient");
  compare_1(expectedd, layer->gradient_bias, 2, "gradient_bias");
  double expectedw1[2][3] = { {6.9891e-04,  1.0139e-01,  2.0208e-01}, {3.0044e-01,  4.0091e-01,  5.0137e-01} };
  update_weights(layer, 0.001, 0.1, 1);
  compare_2(expectedw1, layer->weights, 2, 3, "updated weights");
  double expectedb1[2] = {0.0069891, 0.5047470};
  compare_1(expectedb1, layer->bias, 2, "udpated bias");
  destroy_layer(layer);
  return 0;
}
