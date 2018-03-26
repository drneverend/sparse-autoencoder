#ifndef __LINEAR_ALGEBRA_H
#define __LINEAR_ALGEBRA_H

double dot(double* a, double* b, int n);
void cross(double* a, int la, double* b, int lb, double** result);

void matrix_multi_21(double** a, int rows, int cols, double* b, double* result);

void matrix_add_1(double* a, double* b, double* result, int n);
void matrix_add_2(double** a, double** b, double** result, int rows, int cols);

void multi_22(double** a, int rows, int cols, double** b, double** result);
void multi_11(double* a, double* b, double* result, int n);

void copy_array(double* src, int n, double* dst);

#endif
