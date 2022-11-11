cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)
project(axpy_test CUDA)

add_executable(lab2_test main.cu)

target_link_libraries(lab2_test
                      PRIVATE lab2_axpy_cpu
                      PRIVATE lab2_axpy_openmp
                      PRIVATE lab2_axpy_cuda)

set_target_properties(lab2_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(openmp)
