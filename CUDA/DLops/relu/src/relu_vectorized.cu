
#include <cmath>
#include <cuda_runtime.h>
#include "../include/relu_kernel.h"
__global__
void vector_relu(float4 *input , float4 *output , int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        float4 in = input[idx];
        float4 out;

        out.x = fmaxf(0.0f , in.x);
        out.y = fmaxf(0.0f , in.y);
        out.z = fmaxf(0.0f , in.z);
        out.w = fmaxf(0.0f , in.w);


        output[idx] = out;
    }
}



void launch_relu_vectorized(float * input , float * output , int size){
    int vector_size = size/4;

    int blockSize = 256;

    int gridSize = (vector_size + blockSize -1) / blockSize;

    vector_relu<<<gridSize , blockSize>>>((float4*)input , (float4*)output , vector_size);
    cudaDeviceSynchronize();
    // now remaining elements

    if(size % 4 != 0){
        int remaining = size % 4;
        int offset = size - remaining;
        // naive_relu<<<gridSize, blockSize>>>(input, output + offset, remaining);
        // cudaDeviceSynchronize();

        launch_relu_naive(input , output +offset , remaining);
    }
}