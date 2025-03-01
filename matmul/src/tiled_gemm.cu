#include <cuda_runtime.h>

__global__ void gemm_tiled(float *A, float *B, float *C, 
                          float alpha, float beta, 
                          int M, int N, int K) {
    const int TILE_SIZE = 32;
    
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty; 
    int col = bx * TILE_SIZE + tx; 
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + tx < K)
            s_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;
        
        if (col < N && t * TILE_SIZE + ty < K)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;
        
        __syncthreads();
         
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_gemm_tiled(float *d_A, float *d_B, float *d_C, 
                      float alpha, float beta, 
                      int M, int N, int K) {
    
    const int TILE_SIZE = 32;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();

  
}