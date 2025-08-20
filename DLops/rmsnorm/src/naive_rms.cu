#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

#include "../include/util.cuh"


/*
THIS IS A VERY NAIVE VERSION OF RMSNORM 
HERE WE ASSUME THAT THE VALUE OF K IS LESS THAN blockDim.x 
BECAUSE WE ARE USING BLOCK_REDUCE OP TO FIND THE SUM
*/

__global__ void rmsnorm_naive_kernel(
    half* input_matrix,
    half* output_matrix,
    float g,
    int N , int K
){
    /*
    rms(x) = sqrt(summation(x*x) / total_terms)
    */

    int local_index_x = threadIdx.x;
    int bx = blockIdx.x;

    int global_index_x = blockDim.x * blockIdx.x + threadIdx.x;

    const half eps = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half K_ = __int2half_rn(K);

    __shared__ half smem_variance;

    half val = (global_index_x < N * K) ? input_matrix[global_index_x] : __float2half(0.0f);

    half variance = val * val; 

    variance = block_reduce_sum_fp16_fp16(variance);

    if(local_index_x == 0){
        smem_variance = hrsqrt(variance / K_ + eps);
    }
    __syncthreads();

    if(global_index_x < N * K){
        output_matrix[global_index_x] = (val * smem_variance) * g_;
    }
}


void launch_naive_rms(half* input_matrix,
    half* output_matrix,
    float g,
    int N , int K
){
    
}