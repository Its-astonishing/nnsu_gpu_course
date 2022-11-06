cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)

add_executable(lab2_axpy main.cu axpy_cpu.cu axpy_cuda.cu)

set_target_properties(lab2_axpy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
