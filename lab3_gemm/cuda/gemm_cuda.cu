//
// Created by PC on 11/6/2022.
//
#include <cassert>
#include <iostream>
#include "gemm_cuda.cuh"

__global__ void kernel_gemm16(const float* a,
                          const size_t a_h,
                          const size_t a_w,
                          const float* b,
                          const size_t b_h,
                          const size_t b_w,
                          float* c,
                          float alpha,
                          float betta) {
    const auto current_j = blockIdx.x * blockDim.x + threadIdx.x;
    const auto current_i = blockIdx.y * blockDim.y + threadIdx.y;
    const auto c_w = b_w;

    auto a_at = [&](const auto i, const auto j) -> const float&  {
        auto idx = i * a_w + j;
        return a[idx];
    };

    auto b_at = [&](const auto i, const auto j) -> const float& {
        auto idx = i * b_w + j;
        return b[idx];
    };

    auto c_at = [&](const auto i, const auto j) -> float&  {
        auto idx = i * c_w + j;
        return c[idx];
    };

    float current_value = 0;
    for (size_t k = 0; k < b_h; k++) {
        current_value += a_at(current_i, k) * b_at(k, current_j);
    }

    c_at(current_i, current_j) = alpha * current_value + betta * c_at(current_i, current_j);
}

__global__ void kernel_gemm16_optimized(const float* a,
                          const size_t a_h,
                          const size_t a_w,
                          const float* b,
                          const size_t b_h,
                          const size_t b_w,
                          float* c,
                          float alpha,
                          float betta) {
    const auto c_w = b_w;

    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int block_size = 16;

    __shared__ float a_shared[block_size * block_size];
    __shared__ float b_shared[block_size * block_size];

    auto c_at = [&](const auto i, const auto j) -> float&  {
        auto idx = i * c_w + j;
        return c[idx];
    };

    auto blocks_per_row = (a_w + block_size - 1) / block_size;

    float accum_sum = 0;

    for (size_t current_block = 0; current_block < blocks_per_row; current_block++) {
        auto thread_idx_in_block = blockDim.y * threadIdx.y + threadIdx.x;
        a_shared[thread_idx_in_block] = a[(row * a_w) + (current_block * block_size) + threadIdx.x];
        b_shared[thread_idx_in_block] = b[(current_block * block_size + threadIdx.y) * b_w + col];
        __syncthreads();

        for (size_t k = 0; k < block_size; k++) {
            accum_sum += a_shared[threadIdx.y * blockDim.y + k] * b_shared[k * blockDim.y + threadIdx.x];
        }
        __syncthreads();
    }


    c_at(row, col) = alpha * accum_sum + betta * c_at(row, col);
}


void gemm_cuda16(const float* a,
               const size_t a_h,
               const size_t a_w,
               const float* b,
               const size_t b_h,
               const size_t b_w,
               float* c,
               float alpha,
               float betta) {
    constexpr int block_size = 16;
    assert(a_h % block_size == 0);
    assert(a_w % block_size == 0);
    assert(b_h % block_size == 0);
    assert(b_w % block_size == 0);
    assert(a_w == b_h);

    const auto c_h = a_h;
    const auto c_w = b_w;

    dim3 threads_per_block = dim3(block_size, block_size);
    dim3 grid_size = dim3((c_h + threads_per_block.x - 1) / threads_per_block.x, (c_w + threads_per_block.y - 1) / threads_per_block.y);

    kernel_gemm16<<<grid_size, threads_per_block>>>(a, a_h, a_w, b, b_h, b_w, c, alpha, betta);
}

void gemm_cuda16_optimized(const float* a,
                           const size_t a_h,
                           const size_t a_w,
                           const float* b,
                           const size_t b_h,
                           const size_t b_w,
                           float* c,
                           float alpha,
                           float betta) {
    constexpr int block_size = 16;
    assert(a_h % block_size == 0);
    assert(a_w % block_size == 0);
    assert(b_h % block_size == 0);
    assert(b_w % block_size == 0);
    assert(a_w == b_h);

    const auto c_h = a_h;
    const auto c_w = b_w;

    dim3 threads_per_block = dim3(block_size, block_size);
    dim3 grid_size = dim3((c_h + threads_per_block.x - 1) / threads_per_block.x, (c_w + threads_per_block.y - 1) / threads_per_block.y);

    kernel_gemm16_optimized<<<grid_size, threads_per_block>>>(a, a_h, a_w, b, b_h, b_w, c, alpha, betta);
}

