cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)
project(gemm_gpu CUDA)

add_library(lab3_gemm_cuda STATIC gemm_cuda.cu)
