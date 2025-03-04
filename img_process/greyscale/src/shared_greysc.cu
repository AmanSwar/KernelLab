#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#define BLOCKSIZE 16


__global__ void greyscale_shared(
    const float *input,
    float *output,
    int width,
    int height,
    int channels
){
    __shared__ float s_data[BLOCKSIZE][BLOCKSIZE][3];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    if(x < width && y < height){
        int pos = (y * width + x) * channels;

        s_data[ty][tx][0] = input[pos];
        s_data[ty][tx][1] = input[pos + 1];
        s_data[ty][tx][2] = input[pos + 2];

        __syncthreads();

        float r = s_data[ty][tx][0];
        float g = s_data[ty][tx][1];
        float b = s_data[ty][tx][2];


        output[y * width + x] =  0.299f * r + 0.587f * g + 0.114f * b;

    }
}