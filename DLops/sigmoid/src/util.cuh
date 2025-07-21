#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <cuda_fp16.h>

#include <iostream>
#include <type_traits>
 
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
template <class dtype>
void cudnn_sigmoid(dtype* input , dtype* output , int size , int iter=100){

    int n = 1 , c=1 , h=1;

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    
    auto cudnn_dtype = (std::is_floating_point<dtype>::value) ? (CUDNN_DATA_FLOAT) : (CUDNN_DATA_HALF);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, cudnn_dtype, n, c,
                               h, size);

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


template <class dtype>
inline void init_arr(dtype *a, int N) {
  for (int i = 0; i < N; i++) {
    a[i] = rand();
  }
}



inline bool verify_float(float* kernel_out , float* out , int N , float err = 0.0001){
    for(int i = 0 ; i < N ; i++){
        if(std::abs(kernel_out[i] - out[i]) > err){
            return false;
        }
    }
    return true;
}

inline bool verify_half(half *kernel_out, half *out, int N,
                         half err = 0.0001) {
  for (int i = 0; i < N; i++) {
    if (__habs(kernel_out[i] - out[i]) > err) {
      std::cout << i << std::endl;
      printf("%f \n" , __half2float(kernel_out[i]));
      printf("%f \n" , __half2float(out[i]));

      return false;
    }
  }
  return true;
}

template <class dtype>
void runBenchmark(void(*kernel)(dtype* , dtype* , int) , int N , int iter){

    dtype* i = new dtype[N];
    dtype* o = new dtype[N];
    dtype* cudnn_o = new dtype[N];

    init_arr(i , N);

    dtype* input , *output , *cudnn_output;

    cudaMalloc(&input , N * sizeof(dtype)) ;
    cudaMalloc(&output , N * sizeof(dtype));
    cudaMalloc(&cudnn_output , N * sizeof(dtype));

    cudaMemcpy(input , i , N * sizeof(dtype) , cudaMemcpyHostToDevice);
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

    cudaMemcpy(o , output , N * sizeof(dtype) , cudaMemcpyDeviceToHost);
    cudaMemcpy(cudnn_o ,cudnn_output  , N * sizeof(dtype) , cudaMemcpyDeviceToHost);

    //verify
    bool result;
    if(std::is_floating_point<dtype>::value) {
      // result = verify_float(o, cudnn_o, N);
     }else{
      result = verify_half(o, cudnn_o, N);
     }

    std::cout << result << std::endl;
}
}