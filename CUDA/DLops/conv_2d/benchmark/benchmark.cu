#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Your convolution kernel declarations
#include "../include/conv_kernel.h"

// LibTorch / PyTorch headers
#include <torch/torch.h>
#include <ATen/ATen.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(err)                                                   \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                  << " at line " << __LINE__ << std::endl;                \
        exit(EXIT_FAILURE);                                               \
    }

// -----------------------------------------------------------------------
// Utility to fill an array with random floats
// -----------------------------------------------------------------------
void fill_random(float* arr, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        arr[i] = min_val + r * (max_val - min_val);
    }
}

// -----------------------------------------------------------------------
// Compare two arrays element-wise within a tolerance
// -----------------------------------------------------------------------
bool check_result(const float* ref, const float* test, int size, float eps = 1e-5f) {
    for (int i = 0; i < size; i++) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Benchmark a custom kernel that needs filter_size
// -----------------------------------------------------------------------
void benchmark_kernel_naive(
    void (*kernel)(float*, float*, float*, int, int, int),
    const char* name,
    float* d_input,
    float* d_filter,
    float* d_output,
    float* h_output,
    const float* h_ref,
    int width,
    int height,
    int filter_size,
    int out_size,
    int iterations)
{
    // 1) Warm-up / correctness check
    kernel(d_input, d_output, d_filter, width, height, filter_size);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    bool correct = check_result(h_ref, h_output, out_size);
    std::cout << name << " correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // 2) Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel(d_input, d_output, d_filter, width, height, filter_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << name << " average time: " << (elapsed_ms / iterations) << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// -----------------------------------------------------------------------
// Overload for kernels that do NOT have a filter_size parameter
// -----------------------------------------------------------------------
typedef void (*ConvKernelFn)(float*, float*, float*, int, int);

void benchmark_kernel(
    ConvKernelFn kernel,
    const char* name,
    float* d_input,
    float* d_filter,
    float* d_output,
    float* h_output,
    const float* h_ref,
    int width,
    int height,
    int out_size,
    int iterations)
{
    // 1) Warm-up / correctness check
    kernel(d_input, d_output, d_filter, width, height);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    bool correct = check_result(h_ref, h_output, out_size);
    std::cout << name << " correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // 2) Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel(d_input, d_output, d_filter, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << name << " average time: " << (elapsed_ms / iterations) << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// -----------------------------------------------------------------------
// Benchmark PyTorch's conv2d
// -----------------------------------------------------------------------
void benchmark_torch_conv(
    const float* h_input,
    const float* h_filter,
    const float* h_ref,
    int width,
    int height,
    int filter_size,
    int out_size,
    int iterations)
{
    using namespace torch::indexing;

    // Input shape:  [1, 1, height, width]
    // Filter shape: [1, 1, filter_size, filter_size]
    auto input_options  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto filter_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto t_input = torch::from_blob(
        (void*)h_input,
        {1, 1, height, width},
        input_options
    ).clone().cuda();

    auto t_filter = torch::from_blob(
        (void*)h_filter,
        {1, 1, filter_size, filter_size},
        filter_options
    ).clone().cuda();

    // 1) Warm-up and correctness check
    auto output = torch::nn::functional::conv2d(
        t_input,
        t_filter,
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
    );

    // Move result back to CPU
    auto output_cpu = output.to(torch::kCPU).flatten().contiguous();
    float* out_ptr  = output_cpu.data_ptr<float>();

    // Compare with reference
    bool correct = true;
    for (int i = 0; i < out_size; i++) {
        float diff = std::fabs(out_ptr[i] - h_ref[i]);
        if (diff > 1e-5f) {
            correct = false;
            break;
        }
    }
    std::cout << "PyTorch Conv correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // 2) Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        output = torch::nn::functional::conv2d(
            t_input,
            t_filter,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "PyTorch Conv average time: " << (elapsed_ms / iterations) << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main() {
    // For reproducibility
    srand(1234);

    // Dimensions
    int width  = 256;       // image width
    int height = 256;       // image height
    int filter_size = 3;    // e.g., 3x3 filter
    int out_width  = width  - filter_size + 1; // 254
    int out_height = height - filter_size + 1; // 254
    int out_size   = out_width * out_height;   // 254 * 254 = 64516

    // Number of benchmark iterations
    int iterations = 50;

    // Host allocations
    size_t in_bytes  = width * height * sizeof(float);
    size_t out_bytes = out_size * sizeof(float);
    size_t flt_bytes = filter_size * filter_size * sizeof(float);

    float* h_input  = new float[width * height];
    float* h_filter = new float[filter_size * filter_size];
    float* h_output = new float[out_size];  // for GPU results
    float* h_ref    = new float[out_size];  // reference output

    // Fill input and filter with random values
    fill_random(h_input,  width * height, -1.0f, 1.0f);
    fill_random(h_filter, filter_size * filter_size, -1.0f, 1.0f);

    // -------------------------------------------------------------------
    // Compute reference output using PyTorch on CPU
    // -------------------------------------------------------------------
    {
        auto t_input_cpu = torch::from_blob(h_input, {1, 1, height, width}).clone();
        auto t_filter_cpu = torch::from_blob(h_filter, {1, 1, filter_size, filter_size}).clone();

        auto t_out_cpu = torch::nn::functional::conv2d(
            t_input_cpu,
            t_filter_cpu,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
        );
        // Flatten to get contiguous data
        t_out_cpu = t_out_cpu.flatten().contiguous();

        float* ref_ptr = t_out_cpu.data_ptr<float>();
        for (int i = 0; i < out_size; i++) {
            h_ref[i] = ref_ptr[i];
        }
    }

    // -------------------------------------------------------------------
    // Allocate device memory
    // -------------------------------------------------------------------
    float *d_input = nullptr, *d_filter_dev = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter_dev, flt_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));

    // Copy input and filter to GPU
    CUDA_CHECK(cudaMemcpy(d_input,       h_input,  in_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_dev,  h_filter, flt_bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------
    // Benchmark each kernel
    // -------------------------------------------------------------------
    benchmark_kernel_naive(
        launch_naive_conv,
        "Naive Conv",
        d_input,
        d_filter_dev,
        d_output,
        h_output,
        h_ref,
        width,
        height,
        filter_size,
        out_size,
        iterations
    );

    benchmark_kernel(
        launch_tiled_conv,
        "Tiled Conv",
        d_input,
        d_filter_dev,
        d_output,
        h_output,
        h_ref,
        width,
        height,
        out_size,
        iterations
    );

    benchmark_kernel(
        launch_coal_conv,
        "Coalesced Conv",
        d_input,
        d_filter_dev,
        d_output,
        h_output,
        h_ref,
        width,
        height,
        out_size,
        iterations
    );

    // -------------------------------------------------------------------
    // Benchmark PyTorch Conv2D (GPU)
    // -------------------------------------------------------------------
    benchmark_torch_conv(
        h_input,
        h_filter,
        h_ref,
        width,
        height,
        filter_size,
        out_size,
        iterations
    );

    // -------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter_dev));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;
    delete[] h_ref;

    return 0;
}
