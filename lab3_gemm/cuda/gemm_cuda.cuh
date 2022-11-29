//
// Created by PC on 11/6/2022.
//

#ifndef CUDA_LABS_NNSU_GEMM_CUDA_CUH
#define CUDA_LABS_NNSU_GEMM_CUDA_CUH

void gemm_cuda16(const float* a,
               const size_t a_h,
               const size_t a_w,
               const float* b,
               const size_t b_h,
               const size_t b_w,
               float* c,
               float alpha,
               float betta);

void gemm_cuda16_optimized(const float* a,
                           const size_t a_h,
                           const size_t a_w,
                           const float* b,
                           const size_t b_h,
                           const size_t b_w,
                           float* c,
                           float alpha,
                           float betta);


#endif //CUDA_LABS_NNSU_GEMM_CUDA_CUH
