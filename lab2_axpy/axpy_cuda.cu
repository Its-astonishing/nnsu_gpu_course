//
// Created by PC on 11/6/2022.
//
#include "axpy_cuda.cuh"

__global__ void kernel_saxpy(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy) {
    auto idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < n) {
        y[idx * incy] += alpha * x[idx *incx];
    }
}
__global__ void kernel_daxpy(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy) {
    auto idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < n) {
        y[idx * incy] += alpha * x[idx *incx];
    }
}


void saxpy_cuda(size_t block_size, size_t n, float alpha, float *x, size_t incx, float *y, size_t incy) {
    const size_t grid_size = (n + block_size - 1) / block_size;

    kernel_saxpy<<<grid_size, block_size>>>(n, alpha, x, incx, y, incy);
}

void daxpy_cuda(size_t block_size, size_t n, double alpha, double *x, size_t incx, double *y, size_t incy) {
    const size_t grid_size = (n + block_size - 1) / block_size;

    kernel_daxpy<<<grid_size, block_size>>>(n, alpha, x, incx, y, incy);
}
