
#include <cuda_runtime.h>
#include "../include/reduction_kernels.h"
__global__
void naive_reduction(
    float *input,
    float *output,
    int N
){

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i]: 0;
    __syncthreads();

    for(int s =1 ; s < blockDim.x ; s *=2){
        if(tid % (2 * s) == 0){
            sdata[tid] += sdata[tid+ s];
        }

        __syncthreads();
    }


    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }



}


void launch_naive_reduction(float* d_input, float* d_output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    naive_reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    
    if (blocksPerGrid > 1) {
        launch_naive_reduction(d_output, d_output, blocksPerGrid);
    }
}