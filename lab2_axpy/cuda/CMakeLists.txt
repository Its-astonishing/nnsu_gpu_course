cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)
project(axpy_gpu CUDA)

add_library(lab2_axpy_cuda STATIC axpy_cuda.cu)
