cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
project(gemm_openmp CXX)

find_package(OpenMP)

add_library(lab3_gemm_openmp STATIC gemm_openmp.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab3_gemm_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()
