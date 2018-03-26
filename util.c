#include <stdio.h>
#include <stdlib.h>
#include "util.h"

double** create_2d_array(int rows, int columns) {
  double** result = (double**)malloc(rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    result[i] = (double*)malloc(columns * sizeof(double));
  }
  return result;
}

void destroy_2d_array(double** p, int rows) {
  if (p != NULL) {
    for (int i = 0; i < rows; i++) {
      if (p[i] != NULL) {
        free(p[i]);
      }
    }
    free(p);
  }
}

void zero_matrix_2(double** m, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m[i][j] = 0;
    }
  }
}

void zero_matrix_1(double* m, int n) {
  for (int i = 0; i < n; i++) {
    m[i] = 0;
  }
}

