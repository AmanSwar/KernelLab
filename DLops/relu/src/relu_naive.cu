
#include <cmath>
#include <cuda_runtime.h>
#include "../include/relu_kernel.h"
__global__
void naive_relu(float *input , float *output,int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        //main relu op
        output[idx] = fmaxf(0.0f , input[idx]);
    }

}

void launch_relu_naive(float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    naive_relu<<<gridSize, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
}