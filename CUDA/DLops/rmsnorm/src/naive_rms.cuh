#pragma once


#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  const unsigned int MASK = 0xffffffffu;
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    v += __shfl_xor_sync(MASK, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float *sdata, float val) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int wid = tid / WARP_SIZE; // warp id

  val = warp_reduce_sum_f32(val);

  if (lane == 0)
    sdata[wid] = val;

  __syncthreads();

  float total = 0.0f;
  if (wid == 0) {
    float v = (lane < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? sdata[lane]
                                                                  : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}

__global__
void rms_norm_kernel_bf16(
    const nv_bfloat16* __restrict__ input_matrix_ptr,
    const nv_bfloat16* __restrict__ weight_ptr,
    nv_bfloat16* __restrict__ out_matrix_ptr,
    int M, int N,
    float eps
){
  int row_index = blockIdx.x;
  if (row_index >= M) return;
  int row_start = row_index * N;

  extern __shared__ float sdata[];
  float partial = 0.0f;
  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    float in_f = __bfloat162float(input_matrix_ptr[row_start + idx]);
    partial += in_f * in_f;
  }

  float total_sum = block_reduce_sum_f32(sdata, partial);

  float rms = sqrtf((total_sum / N) + eps);

  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    float in_f = __bfloat162float(input_matrix_ptr[row_start + idx]);
    float w_f = __bfloat162float(weight_ptr[idx]); 
    float out_f = (in_f / rms) * w_f;
    out_matrix_ptr[row_start + idx] = __float2bfloat16(out_f);
  }
}

void launch_rms_bf16(
    const nv_bfloat16 *input_matrix,const nv_bfloat16 *weight_matrix,
    nv_bfloat16 *out_matrix, int M, int N,
    float eps = 1e-6f
){
  int threads_per_block = 256; 
  int blocks_per_grid = M;
  int NUM_WARPS = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;

  size_t smem_size = (threads_per_block + NUM_WARPS) * sizeof(float);
  rms_norm_kernel_bf16<<<blocks_per_grid, threads_per_block, smem_size>>>(
        input_matrix, weight_matrix, out_matrix, M, N, eps
    );

}