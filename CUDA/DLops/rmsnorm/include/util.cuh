#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32


__device__ __forceinline__ half warp_reduce_sum_fp16(half val){
    constexpr int MASK = 0xffffffff;
    #pragma unroll
    for(int offset = WARP_SIZE >> 1 ; offset >= 1 ; offset >>= 1){
        val += __shfl_xor_sync(MASK ,val ,  offset);
    }
    return val;
}


/*
Warp reduce function to accumilate in fp32 for better precision
*/
__device__ __forceinline__ float warp_reduce_sum_fp16_fp32(half val){
    constexpr int MASK = 0xffffffff;

    float val_fp32 = __half2float(val);
    #pragma unroll 
    for(int offset = WARP_SIZE >> 1 ; offset >= 1 ; offset >>= 1){
      val_fp32 += __shfl_xor_sync(MASK, val_fp32, offset);
    }
    return val_fp32;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  constexpr int MASK = 0xffffffff;
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_xor_sync(MASK, val, offset);
  }
  return val;
}


template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_fp16_fp16(half val) {

  int local_index_x = threadIdx.x;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp_id = local_index_x / WARP_SIZE;
  int lane_id = local_index_x % WARP_SIZE;

  static __shared__ float smem[NUM_WARPS];

  float val_fp32 = warp_reduce_sum_fp16_fp32(val);

  if(lane_id == 0){
    smem[warp_id] = val_fp32;
  }

  __syncthreads();

  val_fp32 = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
  val_fp32 = warp_reduce_sum_f32(val_fp32);

  return val_fp32;
}


__device__ __forceinline__ nv_bfloat16 warp_reduce_sum_bf16(nv_bfloat16 val){
  constexpr uint64_t MASK = 0xffffffff;
  #pragma unroll
  for(int offset = MASK >> 1 ; offset >= 1 ; offset >>= 1){
    val += __shfl_xor_sync(MASK , val , offset);
  }
  return val;
}  

//function to reduce sum in a single block -> stored in smem
__device__ __forceinline__ nv_bfloat16 block_reduce_sum_bf16(
  nv_bfloat16* smem_ptr,
  nv_bfloat16* temp_smem_ptr
){
  int idx = threadIdx.x;

  int warp_id = idx / 32;
  int lane_id = idx % 32;

  nv_bfloat16 value = smem_ptr[idx];

  const int NUM_WARPS = (blockDim.x + WARP_SIZE - 1 / WARP_SIZE);

  value = warp_reduce_sum_bf16(value);

  if(lane_id == 0){
    temp_smem_ptr[warp_id] = value;
  }

  __syncthreads();


  value = (lane_id < NUM_WARPS) ? temp_smem_ptr[lane_id] : 0.0f;
  value = warp_reduce_sum_bf16(value);

  return value;

}