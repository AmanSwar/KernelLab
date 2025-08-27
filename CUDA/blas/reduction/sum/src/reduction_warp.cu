
#include <cuda_runtime.h>
#include "../include/reduction_kernels.h"
__global__
void warp_optim_reduction(
    float *input ,
    float *output,
    int N
){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x  * (blockDim.x * 2) + threadIdx.x;


    sdata[tid] = 0;


    //first round of reduction
    if (i < N) sdata[tid] = input[i];
    //add
    if (i + blockIdx.x < N) sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();


    //reduction in shared mem
    for(int s = blockDim.x /2 ; s > 32 ; s >>=1){
        if(tid < 2){
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }
    // last iter
    // unroll
    if (tid < 32) {
        volatile float* smem = sdata;
        #pragma unroll
        for (int s = 32; s > 0; s >>= 1) {
            smem[tid] += smem[tid + s];
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

}



void launch_warp_optimized_reduction(float* d_input, float* d_output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    warp_optim_reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    
    // Handle multi-level reduction for large arrays
    if (blocksPerGrid > 1) {
        launch_warp_optimized_reduction(d_output, d_output, blocksPerGrid);
    }
}