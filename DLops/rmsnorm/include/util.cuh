#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <const int WARP_SIZE = 32>
__device__ __forceinline__ half warp_reduce_sum_fp16(half val){
    const int mask = 0xffffffff;
    #pragma unroll
    for(int offset = WARP_SIZE >> 1 ; WARP_SIZE >= 1 ; offset >>= 1){
        val += __shfl_xor_sync(mask ,val ,  offset);
    }
    return val;
}


