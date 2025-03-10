#include <cuda_runtime.h>
#include "../include/greyscale_kernel.h"


__global__ void grayscale_unrolled(
    const float* input,
    float* output,
    int width,
    int height,
    int channels,
    int pixelsPerThread
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * pixelsPerThread;
    int total_pixel = width * height;

    #pragma unroll
    for(int i = 0 ; i < pixelsPerThread; i++){
        int pixel_idx = idx + i;

        if(pixel_idx < total_pixel){
            int y = pixel_idx / width;
            int x = pixel_idx % width;


            int pos = (y * width + x) * channels;
            float r = input[pos];
            float g = input[pos + 1];
            float b = input[pos + 2];

            output[pixel_idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}


void launch_multipixel(const float *input, float *output, int width, int height, int channels, int pixelsPerThread){
    dim3 threadsPerBlock(32 , 32);
    dim3 numberOfBlocks((width + threadsPerBlock.x -1) / threadsPerBlock.x , (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grayscale_unrolled<<<numberOfBlocks , threadsPerBlock>>>(input , output , width , height , channels , pixelsPerThread);
    cudaDeviceSynchronize();
}
