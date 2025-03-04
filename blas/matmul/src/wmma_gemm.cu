#include <mma.h>
using namespace nvcuda; // WMMA API

#define M 1024
#define N 1024
#define K 1024

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensor_core_gemm(half* A, half* B, float* C, int M , int N, int K) {
    // Block indices
    int bx = blockIdx.x * (BLOCK_SIZE / WMMA_M);
    int by = blockIdx.y * (BLOCK_SIZE / WMMA_N);

    // Thread index within warp
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // Warp Tile Start
    int row = by * WMMA_M;
    int col = bx * WMMA_N;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;

    // Initialize fragment C to zero
    wmma::fill_fragment(fragC, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A and B tiles into WMMA fragments
        wmma::load_matrix_sync(fragA, A + (row * K + k), K);
        wmma::load_matrix_sync(fragB, B + (k * N + col), N);

        // Perform matrix multiplication
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store result back to global memory
    wmma::store_matrix_sync(C + (row * N + col), fragC, N, wmma::mem_row_major);
}
