#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "../include/vec_add_kernel.h"

float* generateRandomData(int size) {
    float* data = new float[size];
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX; // Generate random float between 0 and 1
    }
    return data;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double runBenchmark(void (*kernel)(float*, float*, float*, int), float* a, float* b, float* c, int size, int iterations) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));  // Properly create CUDA events
    CUDA_CHECK(cudaEventCreate(&end));

    double totalTime = 0.0;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernel(a, b, c, size);
        CUDA_CHECK(cudaEventRecord(end, 0));
        CUDA_CHECK(cudaEventSynchronize(end));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
        totalTime += milliseconds;
        
        // Check for kernel execution errors
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    
    return totalTime / iterations; // Return average time in milliseconds
}

bool verifyResults(float* expected, float* result, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(expected[i] - result[i]) > tolerance) {
            std::cout << "Verification failed at index " << i 
                      << ": Expected " << expected[i] 
                      << ", Got " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

double calculateFLOPS(int size, double timeMs) {
    double operations = static_cast<double>(size);
    double seconds = timeMs / 1000.0;
    return operations / seconds;
}

int main() {
    long long int size = 4e+8; 
    int iterations = 100;    

    std::cout << "=== CUDA Vector Addition Benchmark ===" << std::endl;
    std::cout << "Data size: " << size << " elements (" << (size * sizeof(float) / (1024 * 1024)) << " MB)" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl << std::endl;

    srand(42);

    float* h_a = generateRandomData(size);
    float* h_b = generateRandomData(size);
    float* h_c = new float[size];
    float* h_reference = new float[size]; // For verification

    float* d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < size; ++i) {
        h_reference[i] = h_a[i] + h_b[i];
    }

    std::cout << std::left << std::setw(20) << "Kernel" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup" 
              << std::setw(15) << "GFLOPS" 
              << "Verification" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    double timeVecAdd = runBenchmark(launchVecAdd, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool baseVerify = verifyResults(h_reference, h_c, size);
    double baseGFLOPS = calculateFLOPS(size, timeVecAdd) / 1e9; // Convert to GFLOPS
    
    std::cout << std::left << std::setw(20) << "Basic Vector Add" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeVecAdd
              << std::setw(15) << "1.000x" 
              << std::setw(15) << baseGFLOPS
              << (baseVerify ? "PASS" : "FAIL") << std::endl;

    // Run shared memory implementation
    double timeShared = runBenchmark(launchShared, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool sharedVerify = verifyResults(h_reference, h_c, size);
    double sharedGFLOPS = calculateFLOPS(size, timeShared) / 1e9;
    
    std::cout << std::left << std::setw(20) << "Shared Memory" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeShared
              << std::setw(15) << std::fixed << std::setprecision(3) << (timeVecAdd / timeShared) << "x"
              << std::setw(15) << sharedGFLOPS
              << (sharedVerify ? "PASS" : "FAIL") << std::endl;

    // Run coalesced memory access implementation
    double timeCoalesced = runBenchmark(launchCoalesced, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool coalescedVerify = verifyResults(h_reference, h_c, size);
    double coalescedGFLOPS = calculateFLOPS(size, timeCoalesced) / 1e9;
    
    std::cout << std::left << std::setw(20) << "Coalesced Access" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeCoalesced
              << std::setw(15) << std::fixed << std::setprecision(3) << (timeVecAdd / timeCoalesced) << "x"
              << std::setw(15) << coalescedGFLOPS
              << (coalescedVerify ? "PASS" : "FAIL") << std::endl;

    // Uncomment these sections when you implement the other kernel functions
    /*
    // Run tiled implementation
    double timeTiled = runBenchmark(launchTiled, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool tiledVerify = verifyResults(h_reference, h_c, size);
    double tiledGFLOPS = calculateFLOPS(size, timeTiled) / 1e9;
    
    std::cout << std::left << std::setw(20) << "Tiled" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeTiled
              << std::setw(15) << std::fixed << std::setprecision(3) << (timeVecAdd / timeTiled) << "x"
              << std::setw(15) << tiledGFLOPS
              << (tiledVerify ? "PASS" : "FAIL") << std::endl;

    // Run multi-element implementation
    double timeMultiElement = runBenchmark(launchMultiElement, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool multiVerify = verifyResults(h_reference, h_c, size);
    double multiGFLOPS = calculateFLOPS(size, timeMultiElement) / 1e9;
    
    std::cout << std::left << std::setw(20) << "Multi-Element" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeMultiElement
              << std::setw(15) << std::fixed << std::setprecision(3) << (timeVecAdd / timeMultiElement) << "x"
              << std::setw(15) << multiGFLOPS
              << (multiVerify ? "PASS" : "FAIL") << std::endl;

    // Run vectorized implementation
    double timeVectorized = runBenchmark(launchVectorized, d_a, d_b, d_c, size, iterations);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool vectorizedVerify = verifyResults(h_reference, h_c, size);
    double vectorizedGFLOPS = calculateFLOPS(size, timeVectorized) / 1e9;
    
    std::cout << std::left << std::setw(20) << "Vectorized" 
              << std::setw(15) << std::fixed << std::setprecision(3) << timeVectorized
              << std::setw(15) << std::fixed << std::setprecision(3) << (timeVecAdd / timeVectorized) << "x"
              << std::setw(15) << vectorizedGFLOPS
              << (vectorizedVerify ? "PASS" : "FAIL") << std::endl;
    */

    // Print summary of best performance
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "Best time: " << std::min({timeVecAdd, timeShared, timeCoalesced}) << " ms" << std::endl;
    std::cout << "Maximum speedup: " << std::fixed << std::setprecision(2) 
              << (timeVecAdd / std::min({timeVecAdd, timeShared, timeCoalesced})) << "x" << std::endl;
    std::cout << "Maximum GFLOPS: " << std::max({baseGFLOPS, sharedGFLOPS, coalescedGFLOPS}) << std::endl;

    // Free memory on the host
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_reference;

    // Free memory on the device
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaDeviceReset()); // Reset the CUDA device
    return 0;
}