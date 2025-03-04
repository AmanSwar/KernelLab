#include <cuda_runtime.h>
#include "../include/conv_kernel.h"


__global__ void conv_naive(float *input , float *output , float *filter , int width , int height , int filter_size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;


    if(x < width && x < height){
        float sum = 0.0f;
        int filter_half = filter_size / 2;

        for(int i = -filter_half ; i <= filter_half ; i++){
            for(int j = -filter_half ; j <= filter_half ; j++){
                //checking boundary conditions
                int img_x = min(max(x + j , 0) , width -1);
                int img_y = min(max(y + i , 0) , height -1);
                
                
                // sum of multiplied elements
                sum += input[img_y * width + img_x] * filter[(i + filter_half) * filter_size + (j + filter_half)];

            }
        }
        output[y * width + x] = sum;
    }
}


void launch_naive_conv(float *input , float *output , float *filter , int width , int height , int filter_size){
    dim3 blockSize(16 , 16);
    dim3 gridSize((width + blockSize.x -1) / blockSize.x , (height + blockSize.y -1) / blockSize.y);
    conv_naive<<<gridSize , blockSize>>>(input , output , filter , width , height , filter_size);
    cudaDeviceSynchronize();
}