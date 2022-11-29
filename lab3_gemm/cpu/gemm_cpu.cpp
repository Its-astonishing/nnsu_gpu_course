//
// Created by PC on 11/11/2022.
//

#include "gemm_cpu.h"
#include <cassert>
#include <array>
#include <vector>

void gemm_ref_cpu(const float* a,
                  const size_t a_h,
                  const size_t a_w,
                  const float* b,
                  const size_t b_h,
                  const size_t b_w,
                  float* c,
                  float alpha,
                  float betta) {
    assert(a_w == b_h);

    const auto c_h = a_h;
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

    for (size_t i = 0; i < c_h; i++) {
        for (size_t j = 0; j < c_w; j++) {
            float current_value = 0;
            for (size_t k = 0; k < b_h; k++) {
                current_value += a_at(i, k) * b_at(k, j);
            }
            c_at(i, j) = alpha * current_value + betta * c_at(i, j);
        }
    }
}

void matmul(const float* a,
            const float* b,
            float* c,
            const size_t matrix_size,
            int block_size) {
    assert(matrix_size % block_size == 0);


    std::vector<std::vector<float>>  a_block(block_size, std::vector<float>(block_size, 0));
    std::vector<std::vector<float>>  b_block(block_size, std::vector<float>(block_size, 0));

    const size_t block_count = matrix_size / block_size;

    for (size_t i = 0; i < block_count; i++) {
        for (size_t j = 0; j < block_count; j++) {
            for (size_t k = 0; k < block_count; k++) {

                // Fill blocks
                for (size_t ii = 0; ii < block_size; ii++) {
                    for (size_t jj = 0; jj < block_size; jj++) {
                        // a(i, k)
                        auto a_idx = (i * block_size + ii) * matrix_size + (k * block_size + jj);
                        a_block[ii][jj] = a[a_idx];
                        // b(k, j)
                        auto b_idx = (k * block_size + ii) * matrix_size + (j * block_size + jj);
                        b_block[ii][jj] = b[b_idx];
                    }
                }

                // Perform matrix multiplication by blocks
                for (size_t ii = 0; ii < block_size; ii++) {
                    for (size_t jj = 0; jj < block_size; jj++) {
                        for (size_t kk = 0; kk < block_size; kk++) {
                            // c(i,j) = a(i,k) * b(k, j)
                            auto c_idx = (i * block_size + ii) * matrix_size + (j * block_size + jj);
                            c[c_idx] += a_block[ii][kk] * b_block[kk][jj];
                        }
                    }
                }
            }
        }
    }
}
