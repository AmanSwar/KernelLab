#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cuda_runtime.h>
#include <iostream>


#define WARP_SIZE 32

template <const int warp_size = WARP_SIZE>
__device__ float warp_reduce_max(float val) {
  unsigned int mask = 0xFFFFFFFF;

  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val = max(val, __shfl_xor_sync(mask, val, offset));
  }

  return val;
}


template <class dtype>
inline void init(dtype* a , int N){
    for(int i = 0 ; i < N ; i++){
        a[i] = i;
    }
}


inline float max_cpu(float* arr , int N){
    return *std::max_element(arr , arr+N);
}


inline bool verify(float kernel_out , float out , int N){

    if(kernel_out == out) return true;

    return false;
}


template <class dtype>
void reduceBenchmark(void(*kernel)(dtype* , dtype* , int ) , int N , int iter){
    dtype *ha = new dtype[N];
    dtype *hb = new dtype[N];
    
    dtype* a , *b;
    cudaMalloc(&a , N * sizeof(dtype));
    cudaMalloc(&b , N * sizeof(dtype));

    init<dtype>(ha , N);
    cudaMemcpy(a , ha , sizeof(float) * N , cudaMemcpyHostToDevice);

    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    kernel(a , b , N);

    cudaEventRecord(start);
    for(int i = 0 ; i < iter ; i++){
        kernel(a , b , N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms , start , stop);

    std::cout << "Kernel Time : " << ms / iter << std::endl;
    std::cout << "Kernel GFLOP : " << (N*N / (ms / 1000)) / 1e9 << std::endl;

    
    cudaMemcpy(hb , b ,sizeof(dtype) * N, cudaMemcpyDeviceToHost);

    float correct_val = max_cpu(ha, N);

    std::cout << verify(hb[0], correct_val, N) << std::endl;

    
}