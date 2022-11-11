//
// Created by PC on 11/11/2022.
//
#include <iostream>
#include <vector>
#include "cpu/gemm_cpu.h"


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


int main() {
    if (!sanity_check()) {
        std::cout << "Reference program is incorrect!!" << std::endl;
        return 1;
    } else {
        std::cout << "Checks passed!" << std::endl;
    }

    return 0;
}
