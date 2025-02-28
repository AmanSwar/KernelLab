#include "../include/vec_add_kernel.h"
#include <cuda_runtime.h>


__global__
void coalesced_vector_add(float *a , float *b , float *c , int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int gridSize =  blockDim.x * gridDim.x;

    for(; i < n ; i += gridSize){
        c[i] = a[i]  + b[i];
    }

}


void launchCoalesced(float *a, float *b, float *c, int size){
    int blockSize = 256;
    int gridSize = (size + blockSize -1) / blockSize;

    coalesced_vector_add<<<gridSize/4  , blockSize>>>(a, b, c, size);
    cudaDeviceSynchronize();
}   