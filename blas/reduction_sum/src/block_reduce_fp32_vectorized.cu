#include <cuda_runtime.h>
#include "util.cuh"

#define FLOAT4(val) (*reinterpret_cast<float4*>(&(val)))

template <const int NUM_THREADS = 256 >
__global__ void block_all_reduce_f32x4_kernel(
    float *input ,
    float* output,
    int N
){
    int tid = threadIdx.x;
    int global_idx = (blockDim.x * blockIdx.x + threadIdx.x)*4;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    // float4 reg_a = FLOAT4(input[global_idx]);
    float4 reg_a;
    
    
    if (global_idx + 3 < N) {
      reg_a = FLOAT4(input[global_idx]);
    } else {
      // Handle tail end of the data safely.
      reg_a.x = (global_idx < N) ? input[global_idx] : 0.0f;
      reg_a.y = (global_idx + 1 < N) ? input[global_idx + 1] : 0.0f;
      reg_a.z = (global_idx + 2 < N) ? input[global_idx + 2] : 0.0f;
      reg_a.w = (global_idx + 3 < N) ? input[global_idx + 3] : 0.0f;
    }
    float sum = (global_idx + 3 < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if(lane == 0){
        reduce_smem[warp] = sum;
    }

    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;

    if(warp == 0){
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    }
    if(tid == 0){
        atomicAdd(output , sum);
    }
}

void launch_block_all_reduce_fp32x4(float *a, float *b, int N) {
  int block_Dim = 256;
//   int total_threads_req = (N + 3) / 4;
  int grid_Dim = (N + (block_Dim*4) - 1) / (block_Dim*4);
  block_all_reduce_f32x4_kernel<<<grid_Dim, block_Dim>>>(a, b, N);
}

int main() {
  int N = 100000;
  int iter = 100;

  reduceBenchmark<float>(launch_block_all_reduce_fp32x4, N, iter);
}