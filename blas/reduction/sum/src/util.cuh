
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32



template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int mask = kWarpSize >> 1; mask >= 1 ; mask >>=1){
        val += __shfl_xor_sync(0xffffffff , val , mask);

    }
    return val;
}


template <class dtype>
inline void init(dtype* a , int N){
    for(int i = 0 ; i < N ; i++){
        a[i] = rand();
    }
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
    std::cout << "Kernel GFLOP : " << (2*N / (ms / 1000)) / 1e9 << std::endl;

    




    
}