#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "cpu/axpy_cpu.h"
#include "cuda/axpy_cuda.cuh"
#include "openmp/axpy_openmp.h"

template<typename elem_t>
auto get_rand_real() {
    return static_cast<elem_t>(rand()) / static_cast<elem_t>(RAND_MAX);
}

template<typename elem_t>
bool cross_test() {
    constexpr size_t array_size = 12487962;
    std::vector<elem_t> x_cpu(array_size);
    std::vector<elem_t> y_cpu(array_size);

    std::vector<elem_t> y_cuda_dst(array_size);
    elem_t              alpha = static_cast<float>(rand());

    std::generate(x_cpu.begin(), x_cpu.end(), get_rand_real<elem_t>);
    std::generate(y_cpu.begin(), y_cpu.end(), get_rand_real<elem_t>);

    auto x_openmp = x_cpu;
    auto y_openmp = y_cpu;

    elem_t *x_cuda;
    auto res = cudaMalloc(&x_cuda, sizeof(elem_t) * array_size);
    res = cudaMemcpy(x_cuda, x_cpu.data(), sizeof(elem_t) * array_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on GPU..." << std::endl;
        throw;
    }

    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }
    elem_t *y_cuda;
    res = cudaMalloc(&y_cuda, sizeof(elem_t) * array_size);

    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on GPU..." << std::endl;
        throw;
    }

    res = cudaMemcpy(y_cuda, y_cpu.data(), sizeof(elem_t) * array_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

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

    auto assert_close = [](auto first, auto second) {
        const elem_t atol = 0.00001;
        return std::abs(first - second) < atol;
    };

    auto is_equal = std::equal(y_cuda_dst.begin(), y_cuda_dst.end(), y_cpu.begin(), assert_close);
    return is_equal && std::equal(y_cpu.begin(), y_cpu.end(), y_openmp.begin(), assert_close);
}

template <typename elem_t>
double evaluate_openmp(int n) {
    constexpr int run_times = 100;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::vector<elem_t> y(n, 0);

    std::generate(x.begin(), x.end(), get_rand_real<elem_t>);
    std::generate(y.begin(), y.end(), get_rand_real<elem_t>);

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
    constexpr int run_times = 100;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::vector<elem_t> y(n, 0);

    std::generate(x.begin(), x.end(), get_rand_real<elem_t>);
    std::generate(y.begin(), y.end(), get_rand_real<elem_t>);

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
double evaluate_cuda(int n, int block_size) {
    constexpr int run_times = 100;
    elem_t alpha = rand();
    std::vector<elem_t> x(n, 0);
    std::generate(x.begin(), x.end(), get_rand_real<elem_t>);

    elem_t *x_cuda;
    auto res = cudaMalloc(&x_cuda, sizeof(elem_t) * n);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }
    res = cudaMemcpy(x_cuda, x.data(), sizeof(elem_t) * n, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    std::generate(x.begin(), x.end(), get_rand_real<elem_t>);
    elem_t *y_cuda;
    res = cudaMalloc(&y_cuda, sizeof(elem_t) * n);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }
    res = cudaMemcpy(y_cuda, x.data(), sizeof(elem_t) * n, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < run_times; i++) {
        if constexpr(std::is_same_v<float, elem_t>) {
            saxpy_cuda(block_size, n, alpha, x_cuda, 1, y_cuda, 1);
        } else {
            daxpy_cuda(block_size, n, alpha, x_cuda, 1, y_cuda, 1);
        }
    }

    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cudaFree(x_cuda);
    cudaFree(y_cuda);

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

int main()
{
    if (!cross_test<float>()) {
        std::cout << "function for floats doesn't work!!" << std::endl;
        return 1;
    }
    if (!cross_test<double>()) {
        std::cout << "function for doubles doesn't work!!" << std::endl;
        return 1;
    }
    std::cout << "Sanity checks passed!" << std::endl;

    std::vector<int> n_lengths = {50000000, 100000000, 200000000, 300000000, 400000000, 500000000};
    std::vector<int> block_sizes = {8, 16, 32, 64, 128, 256};

    std::cout << "Evaluating cuda saxpy..." << std::endl;
    for (auto blk: block_sizes) {
        for (auto n: n_lengths) {
            printf("[saxpy_cuda : %-3i blocks]/%-9i : %f sec\n", blk, n, evaluate_cuda<float>(n, blk) / 1000000);
        }
    }

    std::cout << "Evaluating cuda daxpy..." << std::endl;
    for (auto blk: block_sizes) {
        for (auto n: n_lengths) {
            if (n > 300000000) {
                // Don't have more memory on gpu than that
                continue;
            }
            printf("[daxpy_cuda : %-3i blocks]/%-9i : %f sec\n", blk, n, evaluate_cuda<double>(n, blk) / 1000000);
        }
    }


    std::cout << "Evaluating openmp saxpy..." << std::endl;
    for (auto n: n_lengths) {
        printf("[saxpy_openmp]/%-9i : %f sec\n", n, evaluate_openmp<float>(n) / 1000000);
    }
    std::cout << "Evaluating openmp daxpy..." << std::endl;
    for (auto n: n_lengths) {
        printf("[daxpy_openmp]/%-9i : %f sec\n", n, evaluate_openmp<double>(n) / 1000000);
    }

    std::cout << "Evaluating cpu saxpy..." << std::endl;
    for (auto n: n_lengths) {
        printf("[saxpy_cpu]/%-9i : %f sec\n", n, evaluate_cpu<float>(n) / 1000000);
    }
    std::cout << "Evaluating cpu daxpy..." << std::endl;
    for (auto n: n_lengths) {
        printf("[daxpy_cpu]/%-9i : %f sec\n", n, evaluate_cpu<double>(n) / 1000000);
    }

    return 0;
}