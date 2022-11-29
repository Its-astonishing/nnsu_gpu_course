//
// Created by PC on 11/11/2022.
//

#ifndef OPENMP_LABS_NNSU_GEMM_OPENMP_H
#define OPENMP_LABS_NNSU_GEMM_OPENMP_H

void gemm_openmp(const float* a,
                  const size_t a_h,
                  const size_t a_w,
                  const float* b,
                  const size_t b_h,
                  const size_t b_w,
                  float* c,
                  float alpha,
                  float betta);

#endif //OPENMP_LABS_NNSU_GEMM_OPENMP_H
