//
// Created by PC on 11/11/2022.
//

#ifndef CUDA_LABS_NNSU_GEMM_REFERENCE_H
#define CUDA_LABS_NNSU_GEMM_REFERENCE_H

void gemm_ref_cpu(const float* a,
                  const size_t a_h,
                  const size_t a_w,
                  const float* b,
                  const size_t b_h,
                  const size_t b_w,
                  float* c,
                  float alpha,
                  float betta);

void matmul(const float* a,
            const float* b,
            float* c,
            const size_t matrix_size,
            int block_size);

#endif //CUDA_LABS_NNSU_GEMM_REFERENCE_H
