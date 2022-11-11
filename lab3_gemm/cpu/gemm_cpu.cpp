//
// Created by PC on 11/11/2022.
//

#include "gemm_cpu.h"
#include <cassert>

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
