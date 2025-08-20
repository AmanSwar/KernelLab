#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/softmax_kernel.h"
#include "util.cuh"

__global__ void softmax_shared(float *input, float *output, int N, int D) {
    extern __shared__ float shared_d[];
    float *s_data = shared_d;              
    float *s_max = &shared_d[blockDim.x];  
    float *s_sum = &shared_d[blockDim.x * 2]; 

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    // Load data into shared memory in chunks
    float thread_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[row * D + i];
        s_data[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    s_max[tid] = thread_max;
    __syncthreads();

    // Parallel reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute exp and sum in parallel
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        if (i < D) {
            float exp_val = expf(s_data[i] - max_val);
            s_data[i] = exp_val; // Reuse shared memory for exp values
            thread_sum += exp_val;
        }
    }
    s_sum[tid] = thread_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    float sum = s_sum[0];   

    // Write output
    for (int i = tid; i < D; i += blockDim.x) {
        if (i < D) {
            output[row * D + i] = s_data[i] / sum;
        }       
    }
}

void launch_shared_memory_softmax(float* d_input, float* d_output, int N, int D) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = N; // One block per row
    // Shared memory: D for data + threadsPerBlock for max + threadsPerBlock for sum
    int sharedMemSize = (D + 2 * threadsPerBlock) * sizeof(float);

    // Check shared memory limit (e.g., 48 KB on older GPUs, 96 KB on newer ones)
    if (sharedMemSize > 49152) { // Adjust based on your GPU's max shared memory
        printf("Error: Shared memory requirement (%d bytes) exceeds limit!\n", sharedMemSize);
        return;
    }

    softmax_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N, D);
    cudaDeviceSynchronize();
}

int main(){
    int N = 4096;
    int M = 2048;
    int iter = 100;
    benchmarkSoftmax<float>(launch_shared_memory_softmax, M, N, iter);
}