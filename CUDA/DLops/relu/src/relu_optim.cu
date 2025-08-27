
#include <cmath>
#include <cuda_runtime.h>
#include "../include/relu_kernel.h"


__global__
void optim_relu(
    const float* __restrict__  input,
    float* __restrict__ output,
    int size
){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(int idx = tid ; tid < size ; idx += stride){
        output[idx] = fmaxf(0.0f , input[idx]);
    }
}


__global__
void coalesced_relu(const float4* __restrict__ input , float4* __restrict__ output , int size){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(int idx = tid ; idx < size ; idx += stride){

        float4 in = input[idx];
        float4 out;


        out.x = fmaxf(0.0f , in.x);
        out.y = fmaxf(0.0f , in.y);
        out.z = fmaxf(0.0f , in.z);
        out.w = fmaxf(0.0f , in.w);


        output[idx] = out;

    }
}

void launch_relu_optimized(float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int minGridSize = 0, gridSize = 0;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
        optim_relu, 0, 0);
    
    // Calculate grid size based on occupancy calculation
    gridSize = (size + blockSize - 1) / blockSize;
    
    // Use vectorized kernel if size is divisible by 4
    if (size >= 4 && size % 4 == 0) {
        int vector_size = size / 4;
        gridSize = (vector_size + blockSize - 1) / blockSize;
        coalesced_relu<<<gridSize, blockSize>>>((float4*)d_input, (float4*)d_output, vector_size);
    } else {
        optim_relu<<<gridSize, blockSize>>>(d_input, d_output, size);
    }
}