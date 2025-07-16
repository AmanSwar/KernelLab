#include <cuda_runtime.h>
#include "util.cuh"

#define FLOAT4(val) (*reinterpret_cast<float4*>(&(val)))

__global__ void sigmoid_fp32x4x4_kernel(
    float* input,
    float* output,
    int N
){
    /*
    So in this kernel we would be manually doing ops upto 4 indices 
    each index would be of float4 vectorized
    global_idx 0 -> 0 , 1, 2,3
    4 -> 4,5,6,7
    8 ->  
    */
    // global_idx -> 0 , 16 , 32
    int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    //locally we will cover
    // 0 -> 0 , 1 ,2 ,3
    // 4 ... , 8 , 12ls
    
    if(global_idx < N){

        #pragma unroll
        for(int i = 0 ; i < 16 ; i += 4){
            
            float4 regA = FLOAT4(input[global_idx + i]);
            float4 regB;

            regB.x = 1 / (1 + __expf(-regA.x));
            regB.y = 1 / (1 + __expf(-regA.y));
            regB.z = 1 / (1 + __expf(-regA.z));
            regB.w = 1 / (1 + __expf(-regA.w));

            FLOAT4(output[global_idx + i]) = regB;

        }

    }

}


void launch_sigmoid_kernel(
    float* input,
    float* output,
    int N
){
    int block_dim = 512;
    int threads_needed = (N + 15 / 16); // one thread takes care of 16 values
    int grid_dim = (threads_needed + block_dim - 1) / block_dim;

    sigmoid_fp32x4x4_kernel<<<grid_dim , block_dim>>>(input , output , N);
    // cudaDeviceSynchronize();
}

int main() {
  int N = 1000000;
  int iter = 100;
  sigmoid::runBenchmark(launch_sigmoid_kernel, N, iter);
}