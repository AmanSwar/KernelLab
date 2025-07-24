
#include <cuda_runtime.h>
#include "../include/reduction_kernels.h"
__global__
void no_divergence_reduction(
    float *input,
    float *output,
    int N
){
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;

    s_data[tid] = 0;

    //load into shared mem

    //load element 1
    if(i < N) s_data[tid] = input[i];
    // load element 2
    if(i + blockDim.x < N) s_data[tid] += input[i + blockDim.x];

    __syncthreads();

    for(int s = blockDim.x / 2 ; s > 0 ; s >>= 2){
        if(tid < s){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = s_data[0];
    }

}

void launch_no_divergence_reduction(float* d_input, float* d_output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    no_divergence_reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    
    // Handle multi-level reduction for large arrays
    if (blocksPerGrid > 1) {
        launch_no_divergence_reduction(d_output, d_output, blocksPerGrid);
    }
}