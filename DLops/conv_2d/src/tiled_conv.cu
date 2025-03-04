#include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cuda_runtime.h>
#include "../include/conv_kernel.h"

#define TILE_WIDTH 16
#define FILTER_SIZE 3


__global__ void conv_tiled(float *input , float *output , float *filter , int width , int height){
    __shared__ float tile[TILE_WIDTH + FILTER_SIZE -1][TILE_WIDTH + FILTER_SIZE -1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    int filter_half = FILTER_SIZE /2;

    //threads for shared ->
    int shared_x = tx + filter_half;
    int shared_y = ty +filter_half;

    // load into shared mem
    if(x < width && y < width){
        tile[shared_y][shared_x] = input[y * width + x];
    }
    else{
        tile[shared_y][shared_x] = 0.0f;
    }

    __syncthreads();


    float sum = 0.0f;

    if(tx < TILE_WIDTH && ty < TILE_WIDTH && x < width && y < height){
        for(int i = -filter_half; i <= filter_half ; i++){
            for(int j = -filter_half ; j <= filter_half ; j++){
                sum += tile[shared_y + i][shared_x + j] * filter[(i + filter_half) * FILTER_SIZE  + (j + filter_half)];
            }
        }

        output[y * width + x] = sum;
    }


}


void launch_tiled_conv(float *input, float *output, float *filter, int width, int height){
    dim3 blockSize(TILE_WIDTH , TILE_WIDTH);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH , (height + TILE_WIDTH -1) / TILE_WIDTH);

    conv_tiled<<<gridSize , blockSize>>>(input , output , filter , width , height);
    cudaDeviceSynchronize();
}