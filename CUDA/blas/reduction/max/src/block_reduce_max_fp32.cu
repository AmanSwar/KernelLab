#include <cfloat>
#include <cuda_runtime.h>
#include "../include/util.cuh"



template <const int NUM_THREADS = 1024> 
__device__ float block_reduce_max(float *arr , int N){

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  __shared__ float warp_max_smem[NUM_WARPS];
  __shared__ float final_block_max;

  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int local_index = threadIdx.x;
  float value = (global_index < N) ? arr[global_index] : -FLT_MAX;

  float max_per_warp = warp_reduce_max(value);

  //now we need to store : for store -> each warp will store 1 value and the first lane of each warp will store that one val 
  int warp_id = local_index / WARP_SIZE; // tells the which warp it is
  int lane_id = local_index % WARP_SIZE; // to find the first lane

  if(lane_id == 0){
    warp_max_smem[warp_id] = max_per_warp;
  }

  __syncthreads();

  value = (local_index < (NUM_WARPS)) ? warp_max_smem[local_index] : -FLT_MAX;

  if(warp_id == 0){
    float block_max = warp_reduce_max(value);

    if(lane_id == 0){
      final_block_max = block_max;
    }
  }

  __syncthreads();

  return final_block_max;

}


__global__ void reduce_kernel_pass1(float*a ,float* b , int N){
  float block_max = block_reduce_max(a, N);

  if(threadIdx.x == 0){
    b[blockIdx.x] = block_max;
  }
}


__global__ void reduce_kernel_pass2(float* b , int N){
  float final_max = block_reduce_max(b, N);

  if(threadIdx.x == 0){
    b[0] = final_max;
  }
}

void launch_block_reduce(float *a, float *b, int N) {
  int block_dim = 1024;
  int grid_dim = (N + block_dim - 1) / block_dim;

  // First pass
  reduce_kernel_pass1<<<grid_dim, block_dim>>>(a, b, N);

  // Second pass (only if more than one block was needed)
  if (grid_dim > 1) {
    reduce_kernel_pass2<<<1, block_dim>>>(b, grid_dim);
  }
}

int main(){
    int N = 100000;
    int iter = 100;

    reduceBenchmark(launch_block_reduce, N , iter);


}