#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda; // WMMA API

// Default dimensions
#define M_DEFAULT 1024
#define N_DEFAULT 1024
#define K_DEFAULT 1024

// Tensor Core parameters
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__global__ void tensor_core_gemm(
    half* A, 
    half* B, 
    float* C, 
    float alpha,
    float beta,
    int M, 
    int N, 
    int K
) {


    int warpM = blockIdx.y * WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
    int warpN = blockIdx.x;

    int rowStart = warpM * WMMA_M;
    int colStart = warpN * WMMA_N;


    if (rowStart >= M || colStart >= N) return;


    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragCTemp;



    wmma::fill_fragment(fragC, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // bounds
        if (k + WMMA_K <= K) {
 

            wmma::load_matrix_sync(fragA, A + (rowStart * K + k), K);
            wmma::load_matrix_sync(fragB, B + (k * N + colStart), N);

            //matumul
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }
        else {

            int remainingK = K - k;
            break;
        }
    }


    if (beta != 0.0f) {
        wmma::load_matrix_sync(fragCTemp, C + (rowStart * N + colStart), N, wmma::mem_row_major);
        

        for (int i = 0; i < fragC.num_elements; i++) {
            fragC.x[i] = alpha * fragC.x[i] + beta * fragCTemp.x[i];
        }
    }
    else {

        for (int i = 0; i < fragC.num_elements; i++) {
            fragC.x[i] = alpha * fragC.x[i];
        }
    }


    wmma::store_matrix_sync(C + (rowStart * N + colStart), fragC, N, wmma::mem_row_major);
}

void launch_tensor_core_gemm(half* d_A, half* d_B, float* d_C, float alpha, float beta, int M, int N, int K) {
    const int WARPS_M = (M + WMMA_M - 1) / WMMA_M;
    const int WARPS_N = (N + WMMA_N - 1) / WMMA_N;
    

    dim3 gridDim(
        WARPS_N,  
        (WARPS_M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK
    );
    

    dim3 blockDim(BLOCK_SIZE);
    

    tensor_core_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}