#include <cuda_runtime.h>
#include "../include/transpose_kernels.h"


__global__ void transposeNaive(float *A, float *B, int N) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N && y < N) {
        B[x * N + y] = A[y * N + x];
    }
}


void launch_naive(float *A , float *B , int N){
    dim3 blockSize(32 , 32);
    dim3 gridSize((N + blockSize.x -1)/ blockSize.x ,(N + blockSize.y -1)/ blockSize.y);
    transposeNaive<<<gridSize , blockSize>>>(A, B, N);
    cudaDeviceSynchronize();
}