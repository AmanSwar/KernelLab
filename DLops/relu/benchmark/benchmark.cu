#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "../include/relu_kernel.h"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }


int main(){

    int N = 100000;

    int iter = 100;

    size_t mem_size = N * sizeof(float);

    float *h_input = new float[N];
    for(size_t i = 0 ; i < N ; i++){
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }



    float *d_input , *d_output;

    CUDA_CHECK(cudaMalloc(&d_input , mem_size));
    CUDA_CHECK(cudaMalloc(&d_output , mem_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, mem_size, cudaMemcpyHostToDevice));

    auto benchmark_kernel = [=](void (*kernel)(float * , float * , int) , const char* name){
        kernel(d_input , d_output , N);
        

        cudaEvent_t start , stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        CUDA_CHECK(cudaEventRecord(start));
        for(int i = 0 ; i < iter ; i++){
            kernel(d_input , d_output , N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << name << " average time: " << (elapsed_ms / iter) << " ms" << std::endl;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };

    benchmark_kernel(launch_relu_naive,"naive ReLU");
    benchmark_kernel(launch_relu_vectorized,"Vecotrized ReLU");
    benchmark_kernel(launch_relu_optimized,"Optimized ReLU");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
}