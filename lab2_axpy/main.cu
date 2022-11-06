#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

#include "axpy_cpu.h"
#include "axpy_cuda.cuh"

template<typename elem_t>
bool sanity_check() {
    constexpr size_t array_size = 15;
    std::vector<elem_t> x_cpu(array_size);
    std::vector<elem_t> y_cpu(array_size);

    std::vector<elem_t> y_cuda_dst(array_size);
    elem_t              alpha = static_cast<float>(rand());

    for (auto val: x_cpu) {
        val = static_cast<elem_t>(rand());
    }

    for (auto val: y_cpu) {
        val = static_cast<elem_t>(rand());
    }

    elem_t *x_cuda;
    cudaMalloc(&x_cuda, sizeof(elem_t) * array_size);
    cudaMemcpy(x_cuda, x_cpu.data(), sizeof(elem_t) * array_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    elem_t *y_cuda;
    cudaMalloc(&y_cuda, sizeof(elem_t) * array_size);
    cudaMemcpy(y_cuda, y_cpu.data(), sizeof(elem_t) * array_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    const int block_size = 32;

    if constexpr (std::is_same_v<elem_t, float>) {
        saxpy_cpu(array_size, alpha, x_cpu.data(), 1, y_cpu.data(), 1);
        saxpy_cuda(block_size, array_size, alpha, x_cuda, 1,y_cuda, 1);
    } else {
        daxpy_cpu(array_size, alpha, x_cpu.data(), 1, y_cpu.data(), 1);
        daxpy_cuda(block_size, array_size, alpha, x_cuda, 1,y_cuda, 1);
    }


    cudaMemcpy(y_cuda_dst.data(), y_cuda, sizeof(int) * array_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(x_cuda);
    cudaFree(y_cuda);

    return y_cuda_dst == y_cpu;
}

int main()
{
    if (!sanity_check<float>()) {
        std::cout << "function for floats doesn't work!!" << std::endl;
        return 1;
    }
    if (!sanity_check<double>()) {
        std::cout << "function for doubles doesn't work!!" << std::endl;
        return 1;
    }
    std::cout << "Sanity checks passed!" << std::endl;

    return 0;
}
