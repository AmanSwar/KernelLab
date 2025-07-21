#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <cuda_runtime.h>
#include "util.cuh"




template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_f32_kernel(float* a, float* b , int N){
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;

    //total number of warps in a block
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    //each will hold the max value
    __shared__ float reduce_smem[NUM_WARPS]; 

    //
    float sum = (idx < N) ? a[idx] : 0.0f;

    //warp info
    int warp = tid / WARP_SIZE; 
    int lane = tid % WARP_SIZE;

    //for each thread calculate cuz using __shfl_xor not sync
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    //store max per warp -> max ele @ lane 0
    if (lane == 0){
        reduce_smem[warp] = sum;
    }

    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;

    if(warp == 0){
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }

    if(tid == 0){
        atomicAdd(b , sum);
    }
}

void launch_block_all_reduce_fp32(float* a , float* b , int N){
    int block_Dim = 256;
    int grid_Dim = (N + block_Dim - 1) / block_Dim;

    block_all_reduce_f32_kernel<<<grid_Dim , block_Dim>>>(a, b, N);

    cudaDeviceSynchronize();
}