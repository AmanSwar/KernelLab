#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>


#include "../include/vec_add_kernel.h"
#include "util.cuh"


//float to float4 conversion via memory address : 
//access mem address -> ask the compiler to treat the mem add as float4 -> return the value
#define FLOAT4(value) (*reinterpret_cast<float4 *>(&(value)))

__global__ void vector_add_f32_vectorized(
    float* a,
    float* b,
    float* c,
    int N
){
    //idx => multiple of 4 (0 , 4 , 8)
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    // if(idx == 0){
    // printf("%f" , a[0]);
    // }

    if (idx < N){
        // access float -> convert and store it in reg
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c; 

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        // store float 4
        FLOAT4(c[idx]) = reg_c;
    }
}


void launchVectorized(float *a , float *b , float *c , int size){
    // std::cout << a[0];
    int blockSize = 512;
    int gridSize = ((size/4) + blockSize - 1) / blockSize;
    vector_add_f32_vectorized<<<gridSize, blockSize>>>(a, b , c, size);
    cudaDeviceSynchronize();
}


int main(){
    int N = 100000;
    int iter = 100;
    vec_add::run_vectorAddBenchmark(launchVectorized, N, iter);

}
