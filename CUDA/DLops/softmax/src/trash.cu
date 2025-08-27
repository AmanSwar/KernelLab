#include <cmath>
#include <cuda_runtime.h>
#include "util.cuh"

__global__ void softmax_fp32_kernel(
    float* input,
    float* output,
    int M, // row size
    int N // col size
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;


    /*
    Main Idea -> 
    one block -> one array
    so each block will compute the softmax of an array

    Safe softmax : e(xi - max(arr)) / sum(e(x - max)all)
    Operations :
    max(arr) -> store in register -> denom term (sum of all exp) -> final softmax
    so its -> block_reduce_max -> block_reduce_sum -> exp(xi) / sum -> output
    */
    if(global_idx * N > M) return;

    /*
    My matrix
    0 -> N along column
    0-> M along rows
    0 1 2 3 ... N-1
    1 -> N
    2 -> 2N
    3
    .
    .
    M-1
    */
    int array_start = global_idx * N; // 0 -> N -> N*2 -> N * 3
    

    extern __shared__ float smem[]; // to store the main array
    __shared__ float smem_sum;


    for(int i = idx ; i < N ; i += blockDim.x){
        //load into smem
        smem[idx] = input[array_start + i];
        __syncthreads();
    }
   

    //find max
    float _max = block_max_fp32x4(smem , N);

    // float _denom = block_all_reduce_f32x4(smem , N);
    //get sum for denom
    block_all_reduce_f32x4(smem , smem_sum ,  N);

    for (int i = idx; i < N; i += blockDim.x) {
        float numerator = expf(smem[idx] - _max);
        output[array_start + i] = numerator / smem_sum;
    }
}

void launch_softmax_kernel(
    float* input,
    float* output,
    int M , int N
){
    int block_dim = 256;
    int grid_dim = M;
    size_t shared_mem_size = N * sizeof(float);
    softmax_fp32_kernel<<<grid_dim , block_dim , shared_mem_size>>>(input , output , M , N);
    cudaDeviceSynchronize();
}

int main(){
    int N = 4096;
    int M = 2048;
    int iter = 100;
    benchmarkSoftmax<float>(launch_softmax_kernel , M , N , iter);
}