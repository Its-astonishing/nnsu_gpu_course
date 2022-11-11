//
// Created by PC on 11/6/2022.
//

#ifndef CUDA_LABS_NNSU_AXPY_OPENMP_H
#define CUDA_LABS_NNSU_AXPY_OPENMP_H

void saxpy_openmp(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy);

void daxpy_openmp(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy);

#endif //CUDA_LABS_NNSU_AXPY_OPENMP_H
