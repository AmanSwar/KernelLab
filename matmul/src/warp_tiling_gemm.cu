#include <cuda_runtime.h>
#include "../include/gemm_kernel.h"

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define TILE_M 32
#define TILE_N 8

__global__ void warp_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    int row = (blockIdx.y * (BLOCK_SIZE / TILE_M) + warpId) * TILE_M;
    int col = (blockIdx.x * (BLOCK_SIZE / TILE_N)) * TILE_N + laneId % TILE_N;

    float regA[TILE_M];
    float regB[TILE_N];
    float regC[TILE_M][TILE_N] = {0.0f}; // Accumulator

    for (int k = 0; k < K; k += TILE_M) {
        // Load A into registers (row major)
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) {
            int a_idx = (row + i) * K + (k + laneId / TILE_N);
            regA[i] = (row + i < M && k + laneId / TILE_N < K) ? A[a_idx] : 0.0f;
        }

        // Load B into registers (column major)
        #pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            int b_idx = (k + laneId / TILE_N) * N + (col + j);
            regB[j] = (k + laneId / TILE_N < K && col + j < N) ? B[b_idx] : 0.0f;
        }

        // Compute partial GEMM
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                regC[i][j] += regA[i] * regB[j];
            }
        }
    }

    // Write back results to C
    #pragma unroll
    for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
            int c_idx = (row + i) * N + (col + j);
            if (row + i < M && col + j < N) {
                C[c_idx] = regC[i][j];
            }
        }
    }
}



void launch_gemm_warp(float *d_A, float *d_B, float *d_C, int M, int N, int K){
    dim3 threadsPerBlock(16 , 16);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1)/ BLOCK_SIZE  , (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    warp_gemm<<<blocksPerGrid , threadsPerBlock>>>(d_A, d_B, d_C, M,  N, K);
    cudaDeviceSynchronize();
}