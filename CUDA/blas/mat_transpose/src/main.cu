#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/transpose_kernels.h"

//std error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define NUM_RUNS 10
#define WARMUP_RUNS 3

//matrix init
void initialize_matrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


//verify -> brute fource
void verify_transpose(float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (abs(A[j * N + i] - B[i * N + j]) > 1e-5) {
                printf("Verification failed at [%d,%d]!\n", i, j);
                return;
            }
        }
    }
    printf("Verification passed!\n");
}



float benchmark_kernel(void (*kernel)(float*, float*, int), 
                      float* d_A, float* d_B, int N, 
                      const char* kernel_name) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        kernel(d_A, d_B, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark runs
    float total_time = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel(d_A, d_B, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;
    }

    float avg_time = total_time / NUM_RUNS;
    printf("%s: Average time = %.3f ms\n", kernel_name, avg_time);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return avg_time;
}

int main() {
    // Matrix sizes to test
    int sizes[] = {512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("\nTesting matrix size: %d x %d\n", N, N);
        
        // Host memory allocation
        float* h_A = (float*)malloc(N * N * sizeof(float));
        float* h_B = (float*)malloc(N * N * sizeof(float));
        
        // Initialize input matrix
        initialize_matrix(h_A, N);
        
        // Device memory allocation
        float *d_A, *d_B;
        CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), 
                            cudaMemcpyHostToDevice));
        
        // Benchmark naive kernel
        benchmark_kernel(launch_naive, d_A, d_B, N, "Naive Kernel");
        
        // Copy result back for verification
        CUDA_CHECK(cudaMemcpy(h_B, d_B, N * N * sizeof(float), 
                            cudaMemcpyDeviceToHost));
        verify_transpose(h_A, h_B, N);
        
        // Benchmark shared memory kernel
        benchmark_kernel(launch_shared, d_A, d_B, N, "Shared Memory Kernel");
        
        // Copy result back for verification
        CUDA_CHECK(cudaMemcpy(h_B, d_B, N * N * sizeof(float), 
                            cudaMemcpyDeviceToHost));
        verify_transpose(h_A, h_B, N);
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        free(h_A);
        free(h_B);
        
        CUDA_CHECK(cudaDeviceReset());
    }
    
    return 0;
}