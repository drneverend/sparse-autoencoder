#include <stdio.h>
#include "dataset.h"

void get_dataset_size(const char* filename, int* n, int* width, int* height) {
  /*we are reading data from a sampled dataset from 
   * CS 294A/W, Winter 2011, so just return the fixed values*/

  *n = 10000;
  *width = 8;
  *height = 8;
}

void read_dataset(const char* filename, double** data, int samples, int size) {
  FILE* f = fopen(filename, "rb");
  char buf[4] = {0};
  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < size; j++) {
      fread(&buf[0], 4, 1, f);
      data[i][j] = *((float*)&buf[0]);
    }
  }
  fclose(f);
}
