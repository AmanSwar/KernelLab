#include <cuda_runtime.h>
#include "../include/vec_add_kernel.h"

__global__
void vectorized_vecadd(float4 *a , float4 *b , float4 *c , int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < n/4){
        float4 a_i = a[i];
        float4 b_i = b[i];
        float4 c_i;

        c_i.x  = a_i.x + b_i.x;
        c_i.y = a_i.y + b_i.y;
        c_i.z = a_i.z + b_i.z;
        c_i.w = a_i.w + b_i.w;

        c[i] = c_i;

    }
}


void launchVectorized(float *a , float *b , float *c , int size){
    int blockSize = 256;
    int gridSize = (size / 4 + blockSize - 1) / blockSize;
    vectorized_vecadd<<<gridSize , blockSize>>>((float4 *)a, (float4*)b, (float4*)c, size);
    cudaDeviceSynchronize();
}


