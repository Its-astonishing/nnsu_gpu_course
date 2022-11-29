//
// Created by PC on 11/11/2022.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "cpu/gemm_cpu.h"
#include "openmp/gemm_openmp.h"
#include "cuda/gemm_cuda.cuh"

auto get_rand_float() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

class test_data {
public:
    static void init(size_t h, size_t w) {
        data.resize(h * w);
        std::generate(data.begin(), data.end(), get_rand_float);
    }

    static std::vector<float> get_data(size_t h, size_t w) {
        auto first = data.begin();
        auto last = data.begin() +  h * w;
        return std::vector<float>(first, last);
    }

private:
    test_data();
    static std::vector<float> data;
};
    std::vector<float> test_data::data{};

bool sanity_check() {
    size_t a_h = 3;
    size_t a_w = 2;
    std::vector<float> a(a_h * a_w, 0.0);

    auto b_h = a_w;
    size_t b_w = 4;
    std::vector<float> b(b_h * b_w, 0.0);

    auto c_h = a_h;
    auto c_w = b_w;
    std::vector<float> c(c_h * c_w, 0.0);

    float alpha = 1.0;
    float betta = 1.0;
    std::fill(a.begin(), a.end(), 1.f);
    std::fill(b.begin(), b.end(), 1.f);

    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);

    // Check
    for (size_t i = 0; i < c_h * c_w; i++) {
        if (c[i] != a_w) {
            return false;
        }
    }

    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);
    for (size_t i = 0; i < c_h * c_w; i++) {
        if (c[i] != 2 * a_w) {
            return false;
        }
    }

    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, 2 * betta);
    for (size_t i = 0; i < c_h * c_w; i++) {
        if (c[i] != 5 * a_w) {
            return false;
        }
    }


    a = {1, 2, 3, 4, 5};
    b = {1,
         2,
         3,
         4,
         5};
    c = {0};

    gemm_ref_cpu(a.data(), 1, a.size(), b.data(), b.size(), 1, c.data(), 2.0, 1.0);

    if (c.front() != 110) {
        return false;
    }

    a = {-1, 4,
         2, 3};
    b = {9, -3,
         6, 1};
    c = {-1, -1,
         -1, -1};

    gemm_ref_cpu(a.data(), 2, 2, b.data(), 2, 2, c.data(), 1.0, 1.0);

    if (c[0] != 14) {
        return false;
    }

    if (c[1] != 6) {
        return false;
    }

    if (c[2] != 35) {
        return false;
    }

    if (c[3] != -4) {
        return false;
    }

    return true;
}

double evaluate_performance_cpp(const size_t a_h, const size_t a_w, const size_t b_h, const size_t b_w) {
    auto a = test_data::get_data(a_h, a_w);
    auto b = test_data::get_data(b_h, b_w);
    auto c = test_data::get_data(a_h, b_w);

    float alpha = 2.412f;
    float betta = 3.14f;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

bool validate(const size_t a_h, const size_t a_w, const size_t b_h, const size_t b_w) {
    auto a = test_data::get_data(a_h, a_w);
    auto b = test_data::get_data(b_h, b_w);
    auto c = test_data::get_data(a_h, b_w);
    std::fill(c.begin(), c.end(), 0);
    auto c_ref = c;

    matmul(a.data(), b.data(), c.data(), a_h, 4);

    gemm_ref_cpu(a.data(), a_h, a_h, b.data(), a_h, a_h, c_ref.data(), 1.0, 1.0);

    return c_ref == c;
}

double evaluate_performance_matmul(const size_t matrix_size, int blk) {
    auto a = test_data::get_data(matrix_size, matrix_size);
    auto b = test_data::get_data(matrix_size, matrix_size);
    auto c = test_data::get_data(matrix_size, matrix_size);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    matmul(a.data(), b.data(), c.data(), matrix_size, blk);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

double evaluate_performance_openmp(const size_t a_h, const size_t a_w, const size_t b_h, const size_t b_w) {
    auto a = test_data::get_data(a_h, a_w);
    auto b = test_data::get_data(b_h, b_w);
    auto c = test_data::get_data(a_h, b_w);
    auto c_ref = c;

    float alpha = 2.412f;
    float betta = 3.14f;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gemm_openmp(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#ifdef DEBUG_MODE
    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c_ref.data(), alpha, betta);

    auto assert_close = [](auto first, auto second) {
        const float atol = 0.00001;
        return std::abs(first - second) < atol;
    };

    if (!std::equal(c.begin(), c.end(), c_ref.begin(), assert_close)) {
        throw;
    }
#endif

    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

double evaluate_performance_cuda(const size_t a_h, const size_t a_w, const size_t b_h, const size_t b_w) {
    auto a = test_data::get_data(a_h, a_w);
    auto b = test_data::get_data(b_h, b_w);
    auto c = test_data::get_data(a_h, b_w);

    std::vector<float> c_dst(c.size(), 0.0);

    float alpha = 2.412f;
    float betta = 3.14f;

    float *a_cuda;
    auto res = cudaMalloc(&a_cuda, sizeof(float) * a.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    float *b_cuda;
    res = cudaMalloc(&b_cuda, sizeof(float) * b.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    float *c_cuda;
    res = cudaMalloc(&c_cuda, sizeof(float) * c.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    res = cudaMemcpy(a_cuda, a.data(), sizeof(float) * a.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    res = cudaMemcpy(b_cuda, b.data(), sizeof(float) * b.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    res = cudaMemcpy(c_cuda, c.data(), sizeof(float) * c.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gemm_cuda16(a_cuda, a_h, a_w, b_cuda, b_h, b_w, c_cuda, alpha, betta);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#ifdef DEBUG_MODE
    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);

    cudaMemcpy(c_dst.data(), c_cuda, sizeof(float) * c.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    auto get_diff = [](auto first, auto second) {
        return std::abs(first - second);
    };

    float max_diff = -1;
    for (int i = 0; i < c.size(); i++) {
        const float atol = 0.001;
        auto diff = get_diff(c[i], c_dst[i]);

        if (diff > atol) {
            throw;
        }

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("[log]: Max absolute difference is %f\n", max_diff);
#endif
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

double evaluate_performance_cuda_optimized(const size_t a_h, const size_t a_w, const size_t b_h, const size_t b_w) {
    auto a = test_data::get_data(a_h, a_w);
    auto b = test_data::get_data(b_h, b_w);
    auto c = test_data::get_data(a_h, b_w);

    std::vector<float> c_dst(c.size(), 0.0);

    float alpha = 2.412f;
    float betta = 3.14f;

    float *a_cuda;
    auto res = cudaMalloc(&a_cuda, sizeof(float) * a.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    float *b_cuda;
    res = cudaMalloc(&b_cuda, sizeof(float) * b.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    float *c_cuda;
    res = cudaMalloc(&c_cuda, sizeof(float) * c.size());
    if (res != cudaError::cudaSuccess) {
        std::cout << "Couldn't allocate memory on gpu" << std::endl;
        throw;
    }

    res = cudaMemcpy(a_cuda, a.data(), sizeof(float) * a.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    res = cudaMemcpy(b_cuda, b.data(), sizeof(float) * b.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    res = cudaMemcpy(c_cuda, c.data(), sizeof(float) * c.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (res != cudaError::cudaSuccess) {
        std::cout << "Memcpy failed" << std::endl;
        throw;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gemm_cuda16_optimized(a_cuda, a_h, a_w, b_cuda, b_h, b_w, c_cuda, alpha, betta);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#ifdef DEBUG_MODE
    gemm_ref_cpu(a.data(), a_h, a_w, b.data(), b_h, b_w, c.data(), alpha, betta);

    cudaMemcpy(c_dst.data(), c_cuda, sizeof(float) * c.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    auto get_diff = [](auto first, auto second) {
        return std::abs(first - second);
    };

    float max_diff = -1;
    for (int i = 0; i < c.size(); i++) {
        const float atol = 0.001;
        auto diff = get_diff(c[i], c_dst[i]);

        if (diff > atol) {
            throw;
        }

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("[log]: Max absolute difference is %f\n", max_diff);
#endif

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return static_cast<double>((std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
}

int main() {
    test_data::init(4096, 4096);
    if (!sanity_check()) {
        std::cout << "Reference program is incorrect!!" << std::endl;
        return 1;
    } else {
        std::cout << "Checks passed!" << std::endl;
    }

    if (!validate(256, 256, 256, 256)) {
        std::cout << "matmul is not working!!" << std::endl;
    } else {
        std::cout << "everything is fine!!" << std::endl;
    }

    std::vector<size_t> matrix_size = {512, 1024, 2048, 3200, 4096};

    std::cout << "Evaluating performance of block mat mul..." << std::endl;
    for (auto block_size: {16, 32})
        for (auto size: matrix_size) {
            printf("[matmul_block_cpu[%i]]/ %4i/ %4i/ %4i/ %4i/ : %f sec\n", block_size, size, size, size, size, evaluate_performance_matmul(size, block_size) / 1000000);
        }

    std::cout << "Evaluating performance of gemm_cpp..." << std::endl;
    for (auto size: matrix_size) {
        if (size >= 4096) {
            break;
        }

        printf("[gemm_ref_cpu]/ %4i/ %4i/ %4i/ %4i/ : %f sec\n", size, size, size, size, evaluate_performance_cpp(size, size, size, size) / 1000000);
    }


    std::cout << "Evaluating performance of gemm_cuda16_optimized..." << std::endl;
    for (auto size: matrix_size) {
        try{
            printf("[gemm_cuda16_shared]/ %4i/ %4i/ %4i/ %4i/ : %f sec\n", size, size / 2, size / 2, size, evaluate_performance_cuda_optimized(size, size / 2, size / 2, size) / 1000000);
        } catch (std::exception &any) {
            std::cout << "Cuda gemm doesn't work!" << std::endl;
        }
    }

    std::cout << "Evaluating performance of gemm_cuda16..." << std::endl;
    for (auto size: matrix_size) {
        try{
            printf("[gemm_cuda16]/ %4i/ %4i/ %4i/ %4i/ : %f sec\n", size, size / 2, size / 2, size, evaluate_performance_cuda(size, size / 2, size / 2, size) / 1000000);
        } catch (std::exception &any) {
            std::cout << "Cuda gemm doesn't work!" << std::endl;
        }
    }

    std::cout << "Evaluating performance of gemm_openmp..." << std::endl;
    for (auto size: matrix_size) {
        try{
            printf("[gemm_openmp]/ %4i/ %4i/ %4i/ %4i/ : %f sec\n", size, size, size, size, evaluate_performance_openmp(size, size, size, size) / 1000000);
        } catch (std::exception &any) {
            std::cout << "OpenMP gemm doesn't work!" << std::endl;
        }
    }

    return 0;
}
