#pragma once

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float v = (lane < numWarps) ? sdata[lane] : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}

__global__ void __launch_bounds__(256, 2)
    rmsnorm_vectorized_kernel(const nv_bfloat162 *__restrict__ input_matrix_ptr,
                              const nv_bfloat162 *__restrict__ weight_ptr,
                              nv_bfloat162 *__restrict__ output_matrix_ptr,
                              int M, int N, float eps) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  if (row >= M)
    return;

  int vcols = (N + 1) / 2;
  int row_start = row * vcols;

  extern __shared__ nv_bfloat162 smem[];
  __shared__ float smem_partial_sum[WARP_SIZE];

  float partial = 0.0f;

  int full_pairs = N / 2;
  bool has_tail = (N % 2) != 0;

  for (int idx = tid; idx < full_pairs; idx += blockDim.x) {
    nv_bfloat162 element = input_matrix_ptr[row_start + idx];
    float fx = __bfloat162float(element.x);
    float fy = __bfloat162float(element.y);
    partial += fx * fx + fy * fy;
    smem[idx] = element;
  }

  if (has_tail) {
    int tail_idx = full_pairs;
    if (tid == 0) {

      nv_bfloat162 element = input_matrix_ptr[row_start + tail_idx];

      float fx = __bfloat162float(element.x);

      partial += fx * fx;

      nv_bfloat162 new_el;

      new_el.x = element.x;
      new_el.y = __float2bfloat16(0.0f);

      smem[tail_idx] = new_el;
    }
  }

  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);

  float inv_rms = rsqrtf((total_sum / float(N)) + eps);

  for (int idx = tid; idx < vcols; idx += blockDim.x) {
    nv_bfloat162 element = smem[idx];
    nv_bfloat162 w = weight_ptr[row_start + idx];
    float fx = __bfloat162float(element.x) * inv_rms;
    float fy = __bfloat162float(element.y) * inv_rms;

    float out_x = fx * __bfloat162float(w.x);
    float out_y = fy * __bfloat162float(w.y);

    nv_bfloat162 store;
    store.x = __float2bfloat16(out_x);
    store.y = __float2bfloat16(out_y);

    output_matrix_ptr[row_start + idx] = store;
  }
}

void launch_rmsnorm_bf16_vectorized(const nv_bfloat16 *input_matrix,
                                    const nv_bfloat16 *weight_matrix,
                                    nv_bfloat16 *out_matrix, int M, int N,
                                    float eps = 1e-6f) {
  int threads_per_block = 256;
  int blocks_per_grid = M;

  int vcols = (N + 1) / 2;
  size_t smem_size = size_t(vcols) * sizeof(nv_bfloat162);

  const nv_bfloat162 *input_matrix_2 =
      reinterpret_cast<const nv_bfloat162 *>(input_matrix);
  const nv_bfloat162 *weight_matrix_2 =
      reinterpret_cast<const nv_bfloat162 *>(weight_matrix);
  nv_bfloat162 *output_matrix_2 = reinterpret_cast<nv_bfloat162 *>(out_matrix);

  rmsnorm_vectorized_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(
      input_matrix_2, weight_matrix_2, output_matrix_2, M, N, eps);
}
