#include <cuda_runtime.h>
#include "../include/vec_add_kernel.h"
#include "util.cuh"

__global__
void naive_vector_add(float *a , float *b , float *c , int n){
    //calculation of index
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    //boundary check
    if(index < n){
        c[index] = a[index] + b[index];
    }

}


void launchVecAdd(float *a, float *b, float *c , int size){
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    naive_vector_add<<<gridSize , blockSize>>>(a , b , c , size);
    cudaDeviceSynchronize();
}

int main() {
  int N = 1000000;
  int iter = 100;
  vec_add::run_vectorAddBenchmark(launchVecAdd, N, iter);
}
