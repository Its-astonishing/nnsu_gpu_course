#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

__global__ void kernel_hello_world()
{
    printf("Hello, world!\n");
}

__global__ void kernel_hello_from_cuda()
{
    auto block_idx = blockIdx.x;
    auto thread_idx = threadIdx.x;
    auto global_idx = block_idx * blockDim.x + thread_idx;
    printf("I am from %i block, %i thread (global index: %i)\n", block_idx, thread_idx, global_idx);
}

__global__ void kernel_modify_array(int *array, size_t n)
{
    auto block_idx = blockIdx.x;
    auto thread_idx = threadIdx.x;
    auto global_idx = block_idx * blockDim.x + thread_idx;

    if (global_idx < n) {
        array[global_idx] += global_idx;
    }
}

void modify_array() {
    constexpr size_t array_size = 15;
    std::vector<int> array(array_size, 1);

    std::cout << "Print array before modifying" << std::endl;
    for (size_t i = 0; i < array.size(); i++) {
        printf("a[%zi] = %i\n", i, array[i]);
    }

    int *cuda_array;
    cudaMalloc(&cuda_array, sizeof(int) * array_size);
    cudaMemcpy(cuda_array, array.data(), sizeof(int) * array_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    const int block_size = 4;
    const int grid_size = (array_size + block_size - 1) / block_size;
    kernel_modify_array<<<grid_size, block_size>>>(cuda_array, array.size());
    cudaMemcpy(array.data(), cuda_array, sizeof(int) * array_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    std::cout << "Print array after modifying" << std::endl;
    for (size_t i = 0; i < array.size(); i++) {
        printf("a[%zi] = %i\n", i, array[i]);
    }


    cudaFree(cuda_array);
}



int main()
{
    kernel_hello_world<<<2, 2>>>();
    cudaDeviceSynchronize();
    kernel_hello_from_cuda<<<2, 2>>>();
    cudaDeviceSynchronize();
    modify_array();
    cudaDeviceSynchronize();

    return 0;
}
