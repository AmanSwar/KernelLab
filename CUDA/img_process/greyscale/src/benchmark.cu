#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../include/greyscale_kernel.h"



//greyscale implementation -> CPU (for checkign)
void rgb_to_grayscale_cpu(const float* input, float* output, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pos = (y * width + x) * channels;
            int outPos = y * width + x;
            
            float r = input[pos];
            float g = input[pos + 1];
            float b = input[pos + 2];
            
            output[outPos] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}


// verify the results
bool verify_results(float* cpu_output, float* gpu_output, int width, int height) {
    // error range
    const float epsilon = 1e-5;
    // traverse the arr
    for (int i = 0; i < width * height; i++) {
        if (fabs(cpu_output[i] - gpu_output[i]) > epsilon) {
            printf("failed at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_output[i], gpu_output[i]);
            return false;
        }
    }
    return true;
}



void benchmark_kernel(int width, int height, int channels, int iterations) {
    size_t input_size = width * height * channels * sizeof(float);
    size_t output_size = width * height * 1 * sizeof(float);
    
    //host
    float *h_input = (float*)malloc(input_size);
    float *h_output_gpu = (float*)malloc(output_size);
    float *h_output_cpu = (float*)malloc(output_size);
    

    // init values ->
    for (int i = 0; i < width * height * channels; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    // benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // launch_naive(d_input , d_output , width , height , channels);
    
    // Benchmark GPU implementation
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_naive(d_input , d_output , width , height , channels);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float gpu_avg_time = gpu_milliseconds / iterations;
    
    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Benchmark 
    clock_t cpu_start = clock();
    for (int i = 0; i < iterations; i++) {
        rgb_to_grayscale_cpu(h_input, h_output_cpu, width, height, channels);
    }

    clock_t cpu_end = clock();
    float cpu_milliseconds = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    float cpu_avg_time = cpu_milliseconds / iterations;
    
    // Verify 
    bool results_match = verify_results(h_output_cpu, h_output_gpu, width, height);
    
    // Print 
    printf("Image dimensions: %d x %d (RGB)\n", width, height);
    printf("Kernel execution time:\n");
    printf("  GPU: %.4f ms (average over %d iterations)\n", gpu_avg_time, iterations);
    printf("  CPU: %.4f ms (average over %d iterations)\n", cpu_avg_time, iterations);
    printf("  Speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results verification: %s\n", results_match ? "PASSED" : "FAILED");
    
    // Calculate throughput
    float pixels_processed = width * height * iterations;
    float gpu_throughput = pixels_processed / (gpu_milliseconds / 1000.0f) / 1000000.0f;  // Mpixels/sec
    float cpu_throughput = pixels_processed / (cpu_milliseconds / 1000.0f) / 1000000.0f;  // Mpixels/sec
    
    printf("Throughput:\n");
    printf("  GPU: %.2f Mpixels/sec\n", gpu_throughput);
    printf("  CPU: %.2f Mpixels/sec\n", cpu_throughput);
    
    // Free memory
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv) {
    // Default 
    int width = 1920;
    int height = 1080;
    // RGB
    int channels = 3;  
    int iterations = 100;
    
    
    // Run benchmark
    benchmark_kernel(width, height, channels, iterations);
    
    // Run additional benchmarks with different image sizes
    printf("\n--- Additional benchmarks ---\n");
    int sizes[][2] = { // provided by claude ->
        {640, 480},    // VGA
        {1280, 720},   // HD
        {3840, 2160}   // 4K
    };
    
    for (int i = 0; i < 3; i++) {
        printf("\n");
        benchmark_kernel(sizes[i][0], sizes[i][1], channels, iterations);
    }
    
    return 0;
}