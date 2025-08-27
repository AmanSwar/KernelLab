#include <cuda_runtime.h>
#include "../include/greyscale_kernel.h"

__global__
void greyscale_naive(
    const float *input,
    float *output,
    int width,
    int height,
    int channels
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //boundary check
    if(x < width && y < height){
        int pos = (y * width + x) *channels;
        int outPos = y * width + x;

        float r = input[pos];
        float g = input[pos + 1];
        float b = input[pos + 2];

        output[outPos] = 0.299f * r + 0.587f*g + 0.114f * b;
    }
}


void launch_naive(const float *input , float *output , int width , int height , int channels){
    dim3 threadsPerBlock(32 , 32);
    dim3 numberOfBlocks((width + threadsPerBlock.x -1) / threadsPerBlock.x , (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    greyscale_naive<<<numberOfBlocks , threadsPerBlock>>>(input , output , width , height , channels);
    cudaDeviceSynchronize();


}