#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath> // For std::abs
#include <cstddef>
#include <iostream>

#define WARP_SIZE 32

__device__ __forceinline__ half warp_reduce_sum_fp16(half val) {
  // This function is correct
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void matrix_reduce_sum_fp16_kernel(
    half *input_matrix, // input matrix of dimension M x N
    half *output_arr,   // output array of dimension M
    int M, int N) {
  // The row this block is responsible for is blockIdx.x
  int row_idx = blockIdx.x;

  // Guard condition: if the row index is out of bounds, the whole block does
  // nothing.
  if (row_idx >= M) {
    return;
  }

  int local_index_x = threadIdx.x;
  extern __shared__ half smem[];

  const int total_warps = blockDim.x / WARP_SIZE;
  const int total_tiles = (N + blockDim.x - 1) / blockDim.x;

  half *smem_warp_sums = &smem[0];           // size: total_warps
  half *smem_tile_sums = &smem[total_warps]; // size: total_tiles

  // Each block processes one row. Calculate the start of that row in global
  // memory.
  half *row_start_ptr = input_matrix + row_idx * N;

  for (int tile = 0; tile < total_tiles; tile++) {
    int col_idx = tile * blockDim.x + local_index_x;

    int warp_id = local_index_x / WARP_SIZE;
    int lane_id = local_index_x % WARP_SIZE;

    // Load value, with bounds check
    half val = (col_idx < N) ? row_start_ptr[col_idx] : __float2half(0.0f);

    // 1. First reduction: within each warp
    val = warp_reduce_sum_fp16(val);

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane_id == 0) {
      smem_warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 2. Second reduction: warp 0 reduces the sums from all other warps
    // Load partial sums from shared memory into the first warp
    val = (local_index_x < total_warps) ? smem_warp_sums[local_index_x]
                                        : __float2half(0.0f);

    if (warp_id == 0) {
      val = warp_reduce_sum_fp16(val);
    }

    // Thread 0 (which is in warp 0) now has the sum for the entire tile.
    // It stores this tile's sum in shared memory.
    if (local_index_x == 0) {
      smem_tile_sums[tile] = val;
    }
    __syncthreads();
  }

  // 3. Final reduction: thread 0 sums the results from all tiles
  if (local_index_x == 0) {
    half final_sum = __float2half(0.0f);
    for (int i = 0; i < total_tiles; i++) {
      final_sum += smem_tile_sums[i];
    }
    // Write the final result to the correct output location
    output_arr[row_idx] = final_sum;
  }
}

void launch_matrix_reduce_sum(half *input_matrix, half *output_arr, int M,
                              int N) {
  // A block size of 1024 is high, 256 or 512 is often a better default.
  // But we'll stick with yours for this example.
  int block_dim = 1024;
  int grid_dim = M; // Correct: one block per row

  const int total_warps_in_block = block_dim / WARP_SIZE;
  const int total_tiles = (N + block_dim - 1) / block_dim;
  const size_t smem_size = sizeof(half) * (total_warps_in_block + total_tiles);

  matrix_reduce_sum_fp16_kernel<<<grid_dim, block_dim, smem_size>>>(
      input_matrix, output_arr, M, N);
}

int main() {
  const int M = 1024;
  const int N = 4096;
  const int iter = 100;

  half *hm = new half[M * N];
  half *harr = new half[M];
  half *hans = new half[M];

  // Corrected Initialization
  for (int i = 0; i < M * N; i++) {
    hm[i] = __float2half(float(i) / (float)(M * N));
  }

  half *dm, *darr;
  cudaMalloc(&dm, sizeof(half) * M * N);
  cudaMalloc(&darr, sizeof(half) * M);
  cudaMemcpy(dm, hm, sizeof(half) * M * N, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Warm-up run
  launch_matrix_reduce_sum(dm, darr, M, N);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iter; i++) {
    launch_matrix_reduce_sum(dm, darr, M, N);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, end);
  std::cout << "Kernel Time : " << ms / iter << " ms" << std::endl;
  // GFLOPs calculation note: reduction is N-1 additions per row.
  double gflops = (double)M * (N - 1) / (ms / iter / 1000.0) / 1e9;
  std::cout << "Kernel GFLOPs : " << gflops << std::endl;

  cudaMemcpy(harr, darr, sizeof(half) * M, cudaMemcpyDeviceToHost);

  // CPU version for verification
  for (int i = 0; i < M; i++) {
    float sum = 0.0f; // Use float for CPU accumulator for better precision
    for (int j = 0; j < N; j++) {
      sum += __half2float(hm[i * N + j]);
    }
    hans[i] = __float2half(sum);
  }

  // Verification
  bool pass = true;
  for (int i = 0; i < M; i++) {
    if (std::abs(__half2float(harr[i]) - __half2float(hans[i])) >
        1e-1) { // Looser tolerance for fp16
      std::cout << "Fail at index " << i << "! GPU: " << __half2float(harr[i])
                << ", CPU: " << __half2float(hans[i]) << std::endl;
      pass = false;
      break;
    }
  }

  if (pass) {
    std::cout << "Pass" << std::endl;
  }

  delete[] hm;
  delete[] harr;
  delete[] hans;
  cudaFree(dm);
  cudaFree(darr);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}