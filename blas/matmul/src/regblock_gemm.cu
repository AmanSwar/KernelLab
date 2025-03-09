#include <cuda_runtime.h>

#define BLOCK_SIZE 32  
#define TILE_WIDTH 8   

__global__ void blocked_gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];



    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;


    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        if (row < M && tile * BLOCK_SIZE + tx < K) {
            Asub[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        } else {
            Asub[ty][tx] = 0.0f;
        }
        

        if (col < N && tile * BLOCK_SIZE + ty < K) {
            Bsub[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        } else {
            Bsub[ty][tx] = 0.0f;
        }

        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += Asub[ty][i] * Bsub[i][tx];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_gemm_regblock(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blocked_gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}