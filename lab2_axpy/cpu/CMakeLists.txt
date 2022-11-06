cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
project(axpy_cpu CXX)

add_library(lab2_axpy_cpu STATIC axpy_cpu.cpp)
