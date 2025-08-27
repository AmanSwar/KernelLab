#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util.cuh"


__global__ void softmax_fp32_kernel(
    float* input,
    float* output,
    const int M , const int N
){
    //shared mem to load the array into
    extern __shared__ float smem[];

    int global_index = blockDim.x * blockIdx.x + threadIdx.x; 
    int row_index = blockIdx.x;
    int local_index = threadIdx.x;
    int row_start = row_index * N;
    
    //load row into smem of one block
    for(int idx = local_index ; idx < N ; idx += blockDim.x){
        smem[idx] = input[row_start + idx];
        
    }
    __syncthreads();
    // printf("alright till here");
    //find max 
    float max  = -FLT_MAX;
    for(int idx = local_index ; idx < N ; idx += blockDim.x){
        max = (max < smem[idx]) ? smem[idx] : max;
    }
    //so now each register is holding the max value
    

    float sum = 0;
    //find sum + make all the elements as element - MAX
    for(int idx = local_index ; idx < N ; idx += blockDim.x){
        float element = expf(smem[idx] - max);
        sum += element;
        smem[idx] = element;
    }

    //final output
    
    for(int idx = local_index ; idx < N ; idx += blockDim.x){
        float ans = smem[idx] / sum;
        output[row_start + idx] = ans;
    }



}


void launch_kernel(
    float* input, 
    float* output,
    const int M , const int N
){

    int block_dim = 256;
    int grid_dim = M;

    size_t size_smem = N * sizeof(float);
    softmax_fp32_kernel<<<grid_dim , block_dim , size_smem>>>(input , output , M , N);

}

int main() {
  int N = 4096;
  int M = 2048;
  int iter = 100;
  benchmarkSoftmax<float>(launch_kernel, M, N, iter);
}