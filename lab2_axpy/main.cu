#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "cpu/axpy_cpu.h"
#include "cuda/axpy_cuda.cuh"
#include "open_mp/axpy_openmp.h"

template<typename elem_t>
bool sanity_check() {
    constexpr size_t array_size = 15;
    std::vector<elem_t> x_cpu(array_size);
    std::vector<elem_t> y_cpu(array_size);

    std::vector<elem_t> y_cuda_dst(array_size);
    elem_t              alpha = static_cast<float>(rand());

    for (auto &val: x_cpu) {
        val = static_cast<elem_t>(rand());
    }

    for (auto &val: y_cpu) {
        val = static_cast<elem_t>(rand());
    }

    auto x_openmp = x_cpu;
    auto y_openmp = y_cpu;

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
        saxpy_openmp(array_size, alpha, x_openmp.data(), 1,y_openmp.data(), 1);
    } else {
        daxpy_cpu(array_size, alpha, x_cpu.data(), 1, y_cpu.data(), 1);
        daxpy_cuda(block_size, array_size, alpha, x_cuda, 1,y_cuda, 1);
        daxpy_openmp(array_size, alpha, x_openmp.data(), 1,y_openmp.data(), 1);
    }


    cudaMemcpy(y_cuda_dst.data(), y_cuda, sizeof(elem_t) * array_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(x_cuda);
    cudaFree(y_cuda);

    auto is_equal = y_cuda_dst == y_cpu;
    return is_equal && y_cpu == y_openmp;
}

template <typename elem_t>
double evaluate_openmp(int n) {
    constexpr int run_times = 10;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::vector<elem_t> y(n, 0);

    std::generate(x.begin(), x.end(), rand);
    std::generate(y.begin(), y.end(), rand);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < run_times; i++) {
        if constexpr(std::is_same_v<float, elem_t>) {
            saxpy_openmp(n, alpha, x.data(), 1, y.data(), 1);
        } else {
            daxpy_openmp(n, alpha, x.data(), 1, y.data(), 1);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / run_times);
}

template <typename elem_t>
double evaluate_cpu(int n) {
    constexpr int run_times = 10;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::vector<elem_t> y(n, 0);

    std::generate(x.begin(), x.end(), rand);
    std::generate(y.begin(), y.end(), rand);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < run_times; i++) {
        if constexpr(std::is_same_v<float, elem_t>) {
            saxpy_cpu(n, alpha, x.data(), 1, y.data(), 1);
        } else {
            daxpy_cpu(n, alpha, x.data(), 1, y.data(), 1);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / run_times);
}

template <typename elem_t>
double evaluate_cuda(int n) {
    // todo: implement
    constexpr int run_times = 10;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::vector<elem_t> y(n, 0);

    std::generate(x.begin(), x.end(), rand);
    std::generate(y.begin(), y.end(), rand);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < run_times; i++) {
        if constexpr(std::is_same_v<float, elem_t>) {
            saxpy_cpu(n, alpha, x.data(), 1, y.data(), 1);
        } else {
            daxpy_cpu(n, alpha, x.data(), 1, y.data(), 1);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / run_times);
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

    int n = 500000000;

    std::cout << "Evaluating openmp saxpy..." << std::endl;
    std::cout << evaluate_openmp<float>(n) / 1000000 << " sec" << std::endl;

    std::cout << "Evaluating openmp daxpy..." << std::endl;
    std::cout << evaluate_openmp<double>(n) / 1000000 << " sec" << std::endl;

    std::cout << "Evaluating CPU saxpy..." << std::endl;
    std::cout << evaluate_cpu<float>(n) / 1000000 << " sec" << std::endl;

    std::cout << "Evaluating CPU daxpy..." << std::endl;
    std::cout << evaluate_cpu<double>(n) / 1000000 << " sec" << std::endl;


    return 0;
}