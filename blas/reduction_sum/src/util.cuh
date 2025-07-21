#pragma once


#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int mask = kWarpSize >> 1; mask >= 1 ; mask >>=1){
        val += __shfl_xor_sync(0xffffffff , val , mask);

    }
    return val;
}