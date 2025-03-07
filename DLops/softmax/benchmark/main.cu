#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../include/softmax_kernel.h"
// #include <torch/torch.h>   


#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // Problem dimensions (N rows, each of length D)
    int N = 4096;
    int D = 4096;
    int iterations = 100;  // Number of iterations for benchmarking

    size_t num_elements = N * D;
    size_t mem_size = num_elements * sizeof(float);

    // Allocate host memory and initialize with random values.
    float* h_input = new float[num_elements];
    for (size_t i = 0; i < num_elements; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory.
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, mem_size));
    CUDA_CHECK(cudaMalloc(&d_output, mem_size));

    // Copy host input to device.
    CUDA_CHECK(cudaMemcpy(d_input, h_input, mem_size, cudaMemcpyHostToDevice));

    // Function to benchmark a kernel.
    auto benchmark_kernel = [=](void (*kernel)(float*, float*, int, int), const char* name) {
        // Warmup run.
        kernel(d_input, d_output, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Create CUDA events.
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Start timing.
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            kernel(d_input, d_output, N, D);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Compute elapsed time.
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << name << " average time: " << (elapsed_ms / iterations) << " ms" << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };

    // Benchmark custom kernels.
    benchmark_kernel(launch_naive_softmax, "Naive Softmax");
    benchmark_kernel(launch_shared_memory_softmax, "Shared Memory Softmax");
    benchmark_kernel(launch_warp_optimized_softmax, "Warp Optimized Softmax");
    benchmark_kernel(launch_block_optimized_softmax, "block Optimized Softmax");
    benchmark_kernel(launch_fused_softmax, "SOTA Optimized Softmax");
    // ---- Benchmark against torch::softmax using libtorch ----

    // // Create a PyTorch tensor on CUDA from the host input.
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor input_tensor = torch::from_blob(h_input, {N, D}, options).clone();
    // torch::Tensor output_tensor;

    // // Warmup run.
    // output_tensor = torch::softmax(input_tensor, /*dim=*/1);
    // cudaDeviceSynchronize();

    // // Create CUDA events for PyTorch softmax timing.
    // cudaEvent_t start, stop;
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));

    // CUDA_CHECK(cudaEventRecord(start));
    // for (int i = 0; i < iterations; i++) {
    //     output_tensor = torch::softmax(input_tensor, 1);
    // }
    // CUDA_CHECK(cudaEventRecord(stop));
    // CUDA_CHECK(cudaEventSynchronize(stop));

    // float elapsed_ms;
    // CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    // std::cout << "Torch Softmax average time: " << (elapsed_ms / iterations) << " ms" << std::endl;

    // CUDA_CHECK(cudaEventDestroy(start));
    // CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;

    return 0;
}
