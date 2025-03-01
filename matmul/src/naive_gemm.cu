#include <cstdlib>
#include <cuda_runtime.h>
#include "../include/gemm_kernel.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)


__global__ void gemm_naive(float *A, float *B, float *C, 
                          float alpha, float beta, 
                          int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[K * i + k] * B[k * N + j];
        }
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
}


void launch_gemm_naive(float *d_A, float *d_B, float *d_C, 
                      float alpha, float beta, 
                      int M, int N, int K) {
    
    
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    gemm_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}
