#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/gemm_kernel.h"
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>

// Utility function to initialize matrices with random values
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}


#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to verify results against cuBLAS
bool verify_results(float *h_C, float *h_C_cublas, int M, int N, float tolerance = 1e-5) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C_cublas[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, h_C[i], h_C_cublas[i]);
            return false;
        }
    }
    return true;
}


struct BenchmarkResult {
    std::string name;
    double execution_time_ms;
    bool correct_result;
};


std::vector<BenchmarkResult> benchmark_gemm(int M, int N, int K, int num_runs = 100) {
    std::vector<BenchmarkResult> results;
    
    // Set alpha and beta values
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C_cublas = (float*)malloc(M * N * sizeof(float));
    
    // Initialize input matrices
    srand(42);  // For reproducibility
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Pointers to function types
    typedef void (*GemmKernelFunc)(float*, float*, float*, float, float, int, int, int);
    typedef void (*GemmWarpFunc)(float*, float*, float*, int, int, int); 
    
   
    cudaMemset(d_C, 0, M * N * sizeof(float));
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < num_runs; run++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, 
                   &alpha, 
                   d_B, N, 
                   d_A, K, 
                   &beta, 
                   d_C, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    results.push_back({"cuBLAS", time_ms, true});
    
    // Setup kernel configurations to test
    struct KernelConfig {
        std::string name;
        void* func;
        bool is_warp;
    };
    
    std::vector<KernelConfig> kernels = {
        {"Naive", (void*)launch_gemm_naive, false},
        {"Tiled", (void*)launch_gemm_tiled, false},
        {"Optimized Tiled", (void*)launch_gemm_optiled, false},
        {"Register Blocked", (void*)launch_gemm_regblock, false},
        {"Warp", (void*)launch_gemm_warp, true},
    };
    
    // Benchmark each kernel
    for (const auto& kernel : kernels) {
        cudaMemset(d_C, 0, M * N * sizeof(float));
        cudaDeviceSynchronize();
        
        start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < num_runs; run++) {
            // if (kernel.is_warp) {
            //     ((GemmWarpFunc)kernel.func)(d_A, d_B, d_C, M, N, K);
            // } else {
            ((GemmKernelFunc)kernel.func)(d_A, d_B, d_C, alpha, beta, M, N, K);
            // }
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
        
        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel %s failed: %s\n", kernel.name.c_str(), cudaGetErrorString(err));
            results.push_back({kernel.name, time_ms, false});
            continue;
        }
        
        // Copy result from device to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Verify results
        bool correct = verify_results(h_C, h_C_cublas, M, N);
        
        results.push_back({kernel.name, time_ms, correct});
    }
    
    // Clean up
    cublasDestroy(handle);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cublas);
    
    return results;
}

// Print results in a nice table
void print_results(const std::vector<BenchmarkResult>& results, int M, int N, int K) {
    std::cout << "===============================================" << std::endl;
    std::cout << "GEMM Benchmark Results (M=" << M << ", N=" << N << ", K=" << K << ")" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << std::left << std::setw(20) << "Implementation" 
              << std::right << std::setw(15) << "Time (ms)" 
            //   << std::setw(15) << "Correct?" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    
    double cublas_time = 0;
    for (const auto& result : results) {
        if (result.name == "cuBLAS") {
            cublas_time = result.execution_time_ms;
            break;
        }
    }
    
    for (const auto& result : results) {
        double speedup = cublas_time / result.execution_time_ms;
        std::cout << std::left << std::setw(20) << result.name 
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << result.execution_time_ms 
                //   << std::setw(15) << (result.correct_result ? "Yes" : "No")
                  << std::setw(15) << std::fixed << std::setprecision(3) << speedup << "x" << std::endl;
    }
    std::cout << "===============================================" << std::endl;
}

int main(int argc, char** argv) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int num_runs = 10;
    
    // Parse command line arguments
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        num_runs = atoi(argv[4]);
    }
    
    // Print benchmark info
    printf("Running GEMM benchmark with M=%d, N=%d, K=%d, runs=%d\n", M, N, K, num_runs);
    
    // Run benchmarks for different matrix sizes
    std::vector<BenchmarkResult> results = benchmark_gemm(M, N, K, num_runs);
    
    // Print results
    print_results(results, M, N, K);
    
    // Run additional benchmarks with different sizes if desired
    std::vector<std::vector<int>> additional_sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };
    
    for (const auto& size : additional_sizes) {
        if (M == size[0] && N == size[1] && K == size[2]) continue; 
        
        printf("\nRunning additional benchmark with M=%d, N=%d, K=%d, runs=%d\n", 
               size[0], size[1], size[2], num_runs);
        
        results = benchmark_gemm(size[0], size[1], size[2], num_runs);
        print_results(results, size[0], size[1], size[2]);
    }
    
    return 0;
}