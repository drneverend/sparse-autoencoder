#ifndef __UTIL_H
#define __UTIL_H


double** create_2d_array(int rows, int columns);
void destroy_2d_array(double** p, int rows);
void zero_matrix_2(double** m, int rows, int cols);
void zero_matrix_1(double* m, int n);

#endif
