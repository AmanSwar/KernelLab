#include <cuda_runtime.h>
#include "../include/conv_kernel.h"

#define FILTER_SIZE 3
#define TILE_WIDTH 16


__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];


__global__ void conv_tiled(float *input , float *output , float *filter , int width , int height){
    __shared__ float tile[TILE_WIDTH +  FILTER_SIZE -1][TILE_WIDTH + FILTER_SIZE -1];


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH  + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    int filter_half = FILTER_SIZE /2;

    int shared_x = tx + filter_half;
    int shared_y = ty +filter_half;


    int input_index = y * width + x;

    // optimzation step -> __ldg
    tile[shared_y][shared_x] = (x < width && y < height) ? __ldg(&input[input_index]) : 0.0f;
    __syncthreads();

    float sum = 0.0f;

    if(tx <  TILE_WIDTH && ty < TILE_WIDTH && x < width && y < height){

        #pragma unroll
        for(int i = -filter_half ; i <= filter_half ; i++){
            
            #pragma unroll
            for(int j = -filter_half ; j <= filter_half ; j++){
                sum += tile[shared_y + i][shared_x + j] * d_filter[(i + filter_half) * FILTER_SIZE + (j + filter_half)];
            }
        }

        output[y * width + x] = sum;
    }

}


void launch_coal_conv(float *input, float *output, float *filter, int width, int height){
    dim3 blockSize(16 , 16);
    dim3 gridSize((width + blockSize.x -1) / blockSize.x , (height + blockSize.y -1) / blockSize.y);
    conv_tiled<<<gridSize , blockSize>>>(input , output , filter , width , height);
    cudaDeviceSynchronize();
}