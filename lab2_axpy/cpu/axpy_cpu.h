//
// Created by PC on 11/6/2022.
//

#ifndef CUDA_LABS_NNSU_AXPY_H
#define CUDA_LABS_NNSU_AXPY_H

void saxpy_cpu(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy);

void daxpy_cpu(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy);

#endif //CUDA_LABS_NNSU_AXPY_H
