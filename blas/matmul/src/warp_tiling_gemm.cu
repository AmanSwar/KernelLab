#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define TILE_M 32
#define TILE_N 8

__global__ void warp_gemm(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K) {
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    int row = (blockIdx.y * (BLOCK_SIZE / TILE_M) + warpId) * TILE_M;
    int col = blockIdx.x * BLOCK_SIZE + laneId % TILE_N;

    float regA[TILE_M];
    float regB[TILE_N];
    float regC[TILE_M][TILE_N] = {0.0f}; 


    for (int k = 0; k < K; k += WARP_SIZE) {

        // Load A into registers (row major)
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) {
            int a_idx = (row + i) * K + (k + laneId % (WARP_SIZE / TILE_N));
            regA[i] = (row + i < M && k + laneId % (WARP_SIZE / TILE_N) < K) ? A[a_idx] : 0.0f;
        }

        // Load B into registers (column major)
        #pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            int b_idx = (k + laneId / TILE_N) * N + (col + j);
            regB[j] = (k + laneId / TILE_N < K && col + j < N) ? B[b_idx] : 0.0f;
        }


        // Compute partial GEMM
        #pragma unroll
        for (int kk = 0; kk < WARP_SIZE; kk++) {
        
            float a_val;
            #pragma unroll
            for (int i = 0; i < TILE_M; i++) {
           
                a_val = __shfl_sync(0xffffffff, regA[i], kk % (WARP_SIZE / TILE_N), WARP_SIZE);
                
                #pragma unroll
                for (int j = 0; j < TILE_N; j++) {
              
                    float b_val = __shfl_sync(0xffffffff, regB[j], kk / TILE_N, WARP_SIZE);
                    regC[i][j] += a_val * b_val;
                }
            }
        }
    }



    // in C
    #pragma unroll
    for (int i = 0; i < TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            int c_idx = (row + i) * N + (col + j);
            if (row + i < M && col + j < N) {
                C[c_idx] = alpha * regC[i][j] + beta * C[c_idx];
            }
        }
    }
}

void launch_gemm_warp(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K) {
    const int warps_per_block = 4; // 4 warps = 128threads
    
    dim3 threadsPerBlock(warps_per_block * WARP_SIZE); 
    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + (BLOCK_SIZE / TILE_M * TILE_M) - 1) / (BLOCK_SIZE / TILE_M * TILE_M)
    );
    
    warp_gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}