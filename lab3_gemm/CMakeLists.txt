cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_STANDARD 17)
project(gemm_test CUDA)

add_executable(lab3_test main.cu)

target_link_libraries(lab3_test
                      PRIVATE lab3_gemm_cuda
                      PRIVATE lab3_gemm_cpu
                      PRIVATE lab3_gemm_openmp)

set_target_properties(lab3_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG_MODE")

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(openmp)
