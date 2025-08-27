#include <cuda_runtime.h>
#include "util.cuh"


__global__ void sigmoid_kernel(
    float* input,
    float* output,
    int N
){

    int global_tid = blockDim.x* blockIdx.x + threadIdx.x;

    if(global_tid < N){
        output[global_tid] = 1 / (1 + __expf(-input[global_tid]));
    }


}


void launch_naive_kernel(
    float* input,
    float* output,
    int N
){

    int block_dim = 1024;
    int grid_dim = (N + block_dim -1) / block_dim;

    sigmoid_kernel<<<grid_dim , block_dim>>>(input , output , N);
    cudaDeviceSynchronize();
}

int main() { 
    int N = 1000000;
    int iter = 100;
    sigmoid::runBenchmark(launch_naive_kernel , N , iter); 

}