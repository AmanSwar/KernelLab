#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util.cuh"

#define HALF2(val) (*reinterpret_cast<half2*>(&(val)))

template <const int NUM_THREAD = 256>
__global__ void block_reduce_fp16x2_kernel(
    half* input,
    half* output,
    int N
){

  int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
  int idx = threadIdx.x;

  constexpr int NUM_WARP = (WARP_SIZE + NUM_THREAD - 1) / WARP_SIZE;
  __shared__ half smem[NUM_WARP];

  int warp_id = global_idx / WARP_SIZE;
  int lane_id = global_idx % WARP_SIZE;

  half2 reg_a;

  if(global_idx + 2 < N){
    reg_a = HALF2(input[global_idx]);
  }
  else{
    reg_a.x = (input[global_idx]);
    reg_a.y = (input[global_idx + 1]);

  }

  float sum = (global_idx + 2 < N) ? (reg_a.x + reg_a.y) : 0.0f;
   

  




}