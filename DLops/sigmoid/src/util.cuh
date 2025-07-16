#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

#include <iostream>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__,          \
             cudaGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUDNN(call)                                                      \
  {                                                                            \
    const cudnnStatus_t status = call;                                         \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      printf("cuDNN Error in %s at line %d: %s\n", __FILE__, __LINE__,         \
             cudnnGetErrorString(status));                                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }


namespace sigmoid{
// template <class dtype>
void cudnn_sigmoid(float* input , float* output , int size , int iter=100){

    int n = 1 , c=1 , h=1;

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,  c,  h, size);

    cudnnActivationDescriptor_t sigmoidDesc;
    cudnnCreateActivationDescriptor(&sigmoidDesc);
    cudnnSetActivationDescriptor(sigmoidDesc , CUDNN_ACTIVATION_SIGMOID , CUDNN_NOT_PROPAGATE_NAN , 0.0);
    
    float alpha = 1.0;
    float beta = 0.0;


    //benchmark
    cudaEvent_t start,  stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warmuprun
    cudnnActivationForward(
        handle,
        sigmoidDesc,
        &alpha,
        tensorDesc,
        input,
        &beta,
        tensorDesc,
        output
    );

    cudaEventRecord(start);
    for(int i = 0 ; i < iter ; i++){
      cudnnActivationForward(handle, sigmoidDesc, &alpha, tensorDesc, input,
                             &beta, tensorDesc, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "CUDNN TIME : " << ms / iter << std::endl;
    std::cout << "CUDNN GFLOP : " << (size / (ms/1000)) / 1e9 << std::endl;
    std::cout << std::endl;


    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroyActivationDescriptor(sigmoidDesc);
    cudnnDestroy(handle);
}

inline void init_arr(float *a, int N) {
  for (int i = 0; i < N; i++) {
    a[i] = rand();
  }
}


inline bool verify(float* kernel_out , float* out , int N , float err = 0.0001){

    for(int i = 0 ; i < N ; i++){
        if(std::abs(kernel_out[i] - out[i]) > err){
            return false;
        }
    }
    return true;


}

void runBenchmark(void(*kernel)(float* , float* , int) , int N , int iter){

    float* i = new float[N];
    float* o = new float[N];
    float* cudnn_o = new float[N];

    init_arr(i , N);

    float* input , *output , *cudnn_output;

    cudaMalloc(&input , N * sizeof(float));
    cudaMalloc(&output , N * sizeof(float));
    cudaMalloc(&cudnn_output , N * sizeof(float));

    cudaMemcpy(input , i , N * sizeof(float) , cudaMemcpyHostToDevice);
    std::cout << "Running cudnn sigmoid" << std::endl;
    cudnn_sigmoid(input , cudnn_output , N , iter=iter);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warpmup run
    kernel(input , output , N);

    cudaEventRecord(start);
    for(int i = 0 ; i < iter ; i++){
        kernel(input , output , N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << "Kernel TIME : " << ms / iter << std::endl;
    std::cout << "Kernel GFLOP : " << (N / (ms / 1000)) / 1e9 << std::endl;

    cudaMemcpy(o , output , N * sizeof(float) , cudaMemcpyDeviceToHost);
    cudaMemcpy(cudnn_o ,cudnn_output  , N * sizeof(float) , cudaMemcpyDeviceToHost);

    //verify
    std::cout << verify(o , cudnn_o , N) << std::endl;
}
}