
double dot(double* a, double* b, int n) {
  double result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

void matrix_multi_21(double** a, int rows, int cols, double* b, double* result) {
  for (int i = 0; i < rows; i++) {
    result[i] = dot(a[i], b, cols);
  }
}

void matrix_add_1(double* a, double* b, double* result, int n) {
  for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
  }
}

void matrix_add_2(double** a, double** b, double** result, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
}

void multi_22(double** a, int rows, int cols, double** b, double** result) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }
}

void multi_11(double* a, double* b, double* result, int n) {
  for (int i = 0; i < n; i++) {
      result[i] = a[i] * b[i];
  }
}

void cross(double* a, int la, double* b, int lb, double** result) {
  for (int i = 0; i < la; i++) {
    for (int j = 0; j < lb; j++) {
      result[i][j] = a[i] * b[j];
    }
  }
}

void copy_array(double* src, int n, double* dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = src[i];
  }
}
