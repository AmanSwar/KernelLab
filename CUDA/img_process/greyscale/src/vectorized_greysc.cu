#include <cuda_runtime.h>
#include "../include/greyscale_kernel.h"
#define BLOCKSIZE 16

__global__ void greyscale_vectorized(
    const float4 *input,
    float *output,
    int width,
    int height
){
    __shared__ float4 s_data[BLOCKSIZE][BLOCKSIZE];


    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    //prefetch the data to shared
    if(x < width && y < height){
        s_data[ty][tx] = input[y * width + x];
    }

    __syncthreads();

    // greyscale conv
    if(x < width && y < height){
        float4 pixel = s_data[ty][tx];
        output[y * width + x] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    }


}



void launch_vectorized(const float4 *input, float *output, int width, int height){
    dim3 blockSize(16 , 16);
    dim3 gridSize((width + blockSize.x -1) / blockSize.x , (height + blockSize.y -1)/ blockSize.y);

    greyscale_vectorized<<<gridSize , blockSize>>>(input , output , width ,height);


}