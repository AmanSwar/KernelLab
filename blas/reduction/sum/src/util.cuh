
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <sys/types.h>

#define WARP_SIZE 32



template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int mask = kWarpSize >> 1; mask >= 1 ; mask >>=1){
        val += __shfl_xor_sync(0xffffffff , val , mask);

    }
    return val;
}

__device__ __forceinline__ half warp_reduce_sum_fp16(half val){
    const uint mask = 0xffffffff;
    #pragma unroll 
    for(int offset = 32 >> 1 ; offset >= 1 ; offset >>= 1){
        val += __shfl_xor_sync(mask , val , offset);
    }

    return val;
}


template <class dtype>
inline void init(dtype* a , int N){
    for(int i = 0 ; i < N ; i++){
        a[i] = i;
    }
}


float inline cpu_sum(float* arr , int N){
    float sum = 0;

    for(int i = 0 ; i < N ; i++){
        sum += arr[i];    
    }
    return sum;
}



template <class dtype>
void reduceBenchmark(void(*kernel)(dtype* , dtype* , int ) , int N , int iter){
    dtype *ha = new dtype[N];
    dtype *hb = new dtype[N];
    
    dtype* a , *b;
    cudaMalloc(&a , N * sizeof(dtype));
    cudaMalloc(&b , N * sizeof(dtype));

    init<dtype>(ha , N);
    float cpu_out = cpu_sum(ha, N);
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

    cudaMemcpy(hb , b , sizeof(float) * N , cudaMemcpyDeviceToHost);

    std::cout << "Kernel Time : " << ms / iter << std::endl;
    std::cout << "Kernel GFLOP : " << (2*N / (ms / iter/ 1000)) / 1e9 << std::endl;

    std::cout << hb[0] << " " << cpu_out << std::endl; 
    std::cout << (hb[0] == cpu_out) << std::endl;



    
}