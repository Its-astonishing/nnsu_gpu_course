cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
project(gemm_cpu CXX)

add_library(lab3_gemm_cpu STATIC gemm_cpu.cpp)
