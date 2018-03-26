#ifndef __DATASET_H
#define __DATASET_H

void get_dataset_size(const char* filename, int* n, int* width, int* height);
void read_dataset(const char* filenae, double*** data);

#endif
