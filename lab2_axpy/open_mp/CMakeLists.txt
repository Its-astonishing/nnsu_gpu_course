cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
project(axpy_cpu CXX)

find_package(OpenMP)

add_library(lab2_axpy_openmp STATIC axpy_openmp.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab2_axpy_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()