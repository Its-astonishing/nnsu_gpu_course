cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)

add_executable(lab1_hello_cuda main.cu)

set_target_properties(lab1_hello_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
