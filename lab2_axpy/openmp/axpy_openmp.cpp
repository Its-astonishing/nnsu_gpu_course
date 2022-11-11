//
// Created by PC on 11/6/2022.
//
#include "axpy_openmp.h"
#include <omp.h>
#include <cstdio>

void saxpy_openmp(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy) {
    int i = 0;
#pragma omp parallel for private(i)
    for (i = 0 ; i < n; i++) {
        y[i * incy] += alpha * x[i *incx];
    }
}

void daxpy_openmp(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy) {
    int i = 0;
#pragma omp parallel for private(i)
    for (i = 0 ; i < n; i++) {
        y[i * incy] += alpha * x[i *incx];
    }
}
