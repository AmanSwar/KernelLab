#pragma once


#include <cmath>
#include <cstdint>
#include <string>
#include <iostream>

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


__device__ __forceinline__ nv_bfloat16 warp_reduce_sum_bf16_fp32(nv_bfloat16 val){
  const unsigned int MASK = 0xffffffff;
  float v = __bfloat162float(val);
  #pragma unroll
  for(int offset = (WARP_SIZE >> 1) ; offset >= 1 ; offset >>= 1){
    v += __shfl_xor_sync(MASK , v , offset);
  }
  return __float2bfloat16(v);
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

  const int NUM_WARPS = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  value = warp_reduce_sum_bf16(value);

  if(lane_id == 0){
    temp_smem_ptr[warp_id] = value;
  }

  __syncthreads();


  value = (lane_id < NUM_WARPS) ? temp_smem_ptr[lane_id] : __float2bfloat16(0.0f);
  value = warp_reduce_sum_bf16(value);

  return value;

}




// ============= BENCHMARK and VERIFY FUNCTION ============

#include <vector>
void rmsnorm_cpu(float *input_matrix, float *weight_matrix,
                 float *output_matrix, int M, int N) {

  for (int i = 0; i < M; i++) {
    float sum = 0.0f;

    for (int j = 0; j < N; j++) {
      float elem = input_matrix[i * N + j];
      sum += (elem * elem);
    }

    float rms = std::sqrt((sum / N));

    for (int j = 0; j < N; j++) {
      output_matrix[i * N + j] =
          (input_matrix[i * N + j] / rms) * weight_matrix[j];
    }
  }
}

void verify(float *kernel_output, float *cpu_output, int M, int N,
            float tolerance = 1e-3) {
  for (int i = 0; i < M * N; i++) {
    if (std::abs(kernel_output[i] - cpu_output[i]) > tolerance) {
      std::cout << "FAIL" << std::endl;
      std::cout << "Error at index " << i
                << ": Kernel out: " << kernel_output[i]
                << " CPU out: " << cpu_output[i]
                << " Diff: " << std::abs(kernel_output[i] - cpu_output[i])
                << std::endl;
      return;
    }
  }

  std::cout << "PASS" << std::endl;
  return;
}

void init(float *matrix, int N) {
  for (int i = 0; i < N; i++) { 
    matrix[i] = 1.0f + static_cast<float>(rand()) / RAND_MAX *
                           99.0f; // Random float between 1 and 100
  }
}


void test(void (*function)(nv_bfloat16 *, nv_bfloat16 *, nv_bfloat16 *, int,
                           int), 
          std::string function_name, int M, int N) {

  float *input_mat = new float[M * N];
  float *out_mat = new float[M * N];
  float *weight = new float[N];

  srand(42); 
  init(weight, N);
  init(input_mat, M * N);

  rmsnorm_cpu(input_mat, weight, out_mat, M, N);

  nv_bfloat16 *da, *dw, *dout;
  cudaMalloc(&da, sizeof(nv_bfloat16) * M * N);
  cudaMalloc(&dw, sizeof(nv_bfloat16) * N);       
  cudaMalloc(&dout, sizeof(nv_bfloat16) * M * N); 

  nv_bfloat16 *ha = new nv_bfloat16[M * N];
  nv_bfloat16 *hw = new nv_bfloat16[N];
  nv_bfloat16 *hout = new nv_bfloat16[M * N];

  for (int i = 0; i < M * N; i++) {
    ha[i] = __float2bfloat16(input_mat[i]);
  }

  for (int i = 0; i < N; i++) {
    hw[i] = __float2bfloat16(weight[i]);
  }

  cudaMemcpy(da, ha, sizeof(nv_bfloat16) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw, sizeof(nv_bfloat16) * N, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaEvent_t start_event, end_event;

  cudaStreamCreate(&stream);
  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);

  function(da, dw, dout, M, N);
  cudaDeviceSynchronize();

  const int num_runs = 100;

  cudaEventRecord(start_event, stream);

  for (int run = 0; run < num_runs; run++) {
    function(da, dw, dout, M, N);
  }

  cudaEventRecord(end_event, stream);
  cudaEventSynchronize(end_event);

  float total_time_ms;
  cudaEventElapsedTime(&total_time_ms, start_event, end_event);
  float time_ms = total_time_ms / num_runs; // Average time per run in ms

  long long total_ops = 3LL * M * N;
  float gflops = (total_ops / (time_ms * 1e6)) * 1000.0f; // GFLOPS

  cudaMemcpy(hout, dout, sizeof(nv_bfloat16) * M * N, cudaMemcpyDeviceToHost);

  float *kernel_output = new float[M * N];
  for (int i = 0; i < M * N; i++) {
    kernel_output[i] = __bfloat162float(hout[i]);
  }

  std::cout << "Function: " << function_name << std::endl;
  std::cout << "Matrix size: " << M << "x" << N << std::endl;
  std::cout << "Time: " << time_ms << " ms" << std::endl;
  std::cout << "GFLOPS: " << gflops << std::endl;

  std::cout << "Verification: ";
  verify(kernel_output, out_mat, M, N, 5e-2); // Relaxed tolerance for bfloat16

  delete[] input_mat;
  delete[] out_mat;
  delete[] weight;
  delete[] ha;
  delete[] hw;
  delete[] hout;
  delete[] kernel_output;

  cudaFree(da);
  cudaFree(dw);
  cudaFree(dout);

  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);
  cudaStreamDestroy(stream);

  std::cout << "----------------------------------------" << std::endl;
}