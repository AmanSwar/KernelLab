#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include "../include/greyscale_kernel.h"



__global__ void greyscale_coal(
    const float *input , 
    float *output , 
    int width,
    int height,
    int channels
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < width * height){
        int y = idx / width;
        int x = idx % width;

        int pos = (y * width + x) * channels;

        float r = input[pos];
        float g = input[pos + 1];
        float b = input[pos + 2];

        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}


void launch_coal(const float *input, float *output, int width, int height, int channels){
    dim3 blockSize(16 , 16);
    dim3 gridSize((width + blockSize.x -1) /blockSize.x , (height + blockSize.y - 1) / blockSize.y);
    
    greyscale_coal<<<blockSize , gridSize>>>(input , output , width , height , channels);
    cudaDeviceSynchronize();
}