/*
Code suggested by claude 3.7 not mine
*/


#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include "../include/greyscale_kernel.h"



__constant__ float c_weights[3] = {0.299f, 0.587f, 0.114f};

__global__ void greyscale_fma(
    const float4 *input,
    float *output,
    int width,
    int height
){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x < width && y < height){
        float4 pixel = input[y * width + x];

        float grey = __fmaf_rn(c_weights[0] , pixel.y , 0.0f);
        grey = __fmaf_rn(c_weights[1] , pixel.y , grey);
        grey = __fmaf_rn(c_weights[2] , pixel.z , grey);

        output[y * width + x] = grey;
    }
}


void launch_fma(const float *input, float *output, int width, int height){
    dim3 blockSize(16 , 16);
    dim3 gridSize((width + blockSize.x -1) / (height + blockSize.y -1) / blockSize.y);
    cudaDeviceSynchronize();
}