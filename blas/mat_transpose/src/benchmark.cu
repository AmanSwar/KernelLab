#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../include/transpose_kernels.h"

#define N 1024  // Matrix dimensions (N x N)

// Helper to fill the matrix with sequential values.
void fillMatrix(float* mat, int n) {
    for (int i = 0; i < n * n; ++i) {
        mat[i] = static_cast<float>(i);
    }
}

// Verify that matrix B is the transpose of matrix A.
bool verifyTranspose(const float* A, const float* B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i * n + j] != B[j * n + i]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Calculate the size in bytes of the matrix.
    int size = N * N;
    size_t bytes = size * sizeof(float);

    // Allocate host memory.
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);

    // Initialize the input matrix.
    fillMatrix(h_A, N);

    // Allocate device memory.
    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    // Copy the host matrix to device.
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Benchmark the Naïve Transpose Kernel ----
    cudaEventRecord(start);
    launch_naive(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // Copy the result back to host and verify correctness.
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    if (verifyTranspose(h_A, h_B, N)) {
        std::cout << "Naïve kernel: Transpose is correct." << std::endl;
    } else {
        std::cout << "Naïve kernel: Transpose is incorrect!" << std::endl;
    }
    std::cout << "Naïve kernel execution time: " << ms_naive << " ms" << std::endl;

    // ---- Benchmark the Shared-Memory Transpose Kernel ----
    // Reset device matrices.
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_B, 0, bytes);

    cudaEventRecord(start);
    launch_shared(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start, stop);

    // Copy the result back to host and verify correctness.
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    if (verifyTranspose(h_A, h_B, N)) {
        std::cout << "Shared memory kernel: Transpose is correct." << std::endl;
    } else {
        std::cout << "Shared memory kernel: Transpose is incorrect!" << std::endl;
    }
    std::cout << "Shared memory kernel execution time: " << ms_shared << " ms" << std::endl;

    // Cleanup.
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
