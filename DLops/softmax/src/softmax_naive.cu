#include <cmath>
#include <cuda_runtime.h>
#include "../include/softmax_kernel.h"
__global__ void softmax_naive(
    float *input , float* output , int N , int D
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < N){
        float max_val = -INFINITY;
        for(int i = 0 ; i < D ; i++){
            max_val = fmaxf(max_val , input[tid * D + i]);
        }

        float sum = 0.0f;
        for(int i = 0; i < D ; i++){
            float exp_val = expf(input[tid * D + i] - max_val);
            output[tid * D + i] = exp_val;
            sum += exp_val;
        }

        for (int i = 0; i < D; i++) {
            output[tid * D + i] /= sum;
        }

    }   
}

void launch_naive_softmax(float* d_input, float* d_output, int N, int D) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, D);
    cudaDeviceSynchronize();

}