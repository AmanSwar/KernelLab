
#include "../include/relu_kernel.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#define NUM_ELEMENTS_PER_THREAD 2

__global__ void relu_fp32x16_kernel(
    float4* input_arr,
    float4* output_arr,
    const int N
){
    int global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll
    for (int idx = 0; idx < NUM_ELEMENTS_PER_THREAD; idx++) {
        int element_idx = global_index_x * NUM_ELEMENTS_PER_THREAD + idx;
        
        // Bounds check
        if (element_idx >= N) break;
        
        float4 curr_element = __ldg(&input_arr[element_idx]);
        float4 out_element;
        out_element.x = fmaxf(0.0f, curr_element.x);
        out_element.y = fmaxf(0.0f, curr_element.y);
        out_element.z = fmaxf(0.0f, curr_element.z);
        out_element.w = fmaxf(0.0f, curr_element.w);

        output_arr[element_idx] = out_element;
    }
}

void launch_relu_fp32x16(
    float* input,
    float* output,
    const int N
){
    int block_dim = 256;  
    int elements_per_block = block_dim * NUM_ELEMENTS_PER_THREAD;
    
    int grid_dim = (N + elements_per_block - 1) / elements_per_block;

    relu_fp32x16_kernel<<<grid_dim, block_dim>>>((float4*)input, (float4*)output, N);
    

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}



