#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Include your custom kernel header
#include "../include/relu_kernel.h"

// Include PyTorch headers (assuming LibTorch is installed)
// Adjust these includes as needed based on your LibTorch install path.
#include <torch/torch.h>
#include <ATen/ATen.h>

#define CUDA_CHECK(err)                                            \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)     \
                  << " at line " << __LINE__ << std::endl;         \
        exit(EXIT_FAILURE);                                        \
    }

//--------------------------------------------------------------------------------------
// CPU reference ReLU to validate correctness
//--------------------------------------------------------------------------------------
void relu_cpu(const float* in, float* out, int N)
{
    for (int i = 0; i < N; i++) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

//--------------------------------------------------------------------------------------
// Compare two arrays element-wise within a tolerance
//--------------------------------------------------------------------------------------
bool check_result(const float* ref, const float* test, int N, float eps = 1e-5f)
{
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

//--------------------------------------------------------------------------------------
// Benchmark function for your custom CUDA kernels
//--------------------------------------------------------------------------------------
void benchmark_kernel(void (*kernel)(float*, float*, int),
                      const char* name,
                      float* d_input,
                      float* d_output,
                      float* h_output,
                      float* h_ref,
                      int N,
                      int iter)
{
    // Run kernel once to produce an output to check correctness
    kernel(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check correctness
    bool correct = check_result(h_ref, h_output, N);
    std::cout << name << " correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Now benchmark with multiple iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        kernel(d_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << name << " average time: " << (elapsed_ms / iter) << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

//--------------------------------------------------------------------------------------
// Benchmark function for PyTorch's ReLU (GPU)
//--------------------------------------------------------------------------------------
void benchmark_torch_relu(const float* h_input, float* h_ref, int N, int iter)
{
    // Create a tensor from raw host data; clone() to get a distinct tensor.
    // Then move to GPU (cuda).
    // NOTE: from_blob() doesn't own the data; often you want a clone().
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto t_input = torch::from_blob((void*)h_input, {N}, options).clone().cuda();

    // For correctness check: run ReLU once
    auto t_output = torch::relu(t_input);
    // Move result back to CPU
    auto t_output_cpu = t_output.to(torch::kCPU);

    // Check correctness vs. reference
    bool correct = true;
    auto t_acc = t_output_cpu.accessor<float, 1>();
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(h_ref[i] - t_acc[i]);
        if (diff > 1e-5f) {
            correct = false;
            break;
        }
    }
    std::cout << "PyTorch ReLU correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Now measure performance with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        // out-of-place ReLU; you could do in-place relu_() if desired
        t_output = torch::relu(t_input);
    }
    // Force kernel completion
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "PyTorch ReLU average time: " << (elapsed_ms / iter) << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

//--------------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------------
int main()
{
    int N = 100000;
    int iter = 100;

    size_t mem_size = N * sizeof(float);

    // Allocate host memory for input
    float* h_input = new float[N];
    // Host memory for CPU reference
    float* h_ref   = new float[N];
    // Host memory to copy back results from GPU
    float* h_output = new float[N];

    // Initialize random input
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // some negative, some positive
    }

    // Prepare a CPU reference for correctness checking
    relu_cpu(h_input, h_ref, N);

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, mem_size));
    CUDA_CHECK(cudaMalloc(&d_output, mem_size));

    // Copy input to GPU once
    CUDA_CHECK(cudaMemcpy(d_input, h_input, mem_size, cudaMemcpyHostToDevice));

    // Benchmark each kernel
    benchmark_kernel(launch_relu_naive,       "Naive ReLU",       d_input, d_output, h_output, h_ref, N, iter);
    benchmark_kernel(launch_relu_vectorized,  "Vectorized ReLU",  d_input, d_output, h_output, h_ref, N, iter);
    benchmark_kernel(launch_relu_optimized,   "Optimized ReLU",   d_input, d_output, h_output, h_ref, N, iter);

    // Benchmark PyTorch ReLU
    benchmark_torch_relu(h_input, h_ref, N, iter);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_ref;
    delete[] h_output;

    return 0;
}
