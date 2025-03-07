#include <cuda_runtime.h>
#include "../include/softmax_kernel.h"


__global__ void block_optimized_softmax(float* input, float* output, int N, int D) {
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = &shared_data[blockDim.x];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < N) {
        // Initialize shared memory
        shared_max[tid] = -INFINITY;
        shared_sum[tid] = 0.0f;
        
        // Each thread finds local max in its assigned elements
        for (int i = tid; i < D; i += blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], input[row * D + i]);
        }
        __syncthreads();
        
        // Reduce to find the max value for the row
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
            }
            __syncthreads();
        }
        
        float max_val = shared_max[0];
        __syncthreads();
        
        // Calculate exp and sum
        for (int i = tid; i < D; i += blockDim.x) {
            float exp_val = expf(input[row * D + i] - max_val);
            output[row * D + i] = exp_val;
            shared_sum[tid] += exp_val;
        }
        __syncthreads();
        
        // Reduce to find the sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
        
        float sum = shared_sum[0];
        __syncthreads();
        
        // Normalize
        for (int i = tid; i < D; i += blockDim.x) {
            output[row * D + i] /= sum;
        }
    }
}

// Launch parameters example
void launch_block_optimized_softmax(float* d_input, float* d_output, int N, int D) {
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    int sharedMemSize = 2 * threadsPerBlock * sizeof(float); // For max and sum
    
    block_optimized_softmax<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N, D);
}