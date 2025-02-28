#include <cuda_runtime.h>


__global__
void vectorAdd_shared(float *a , float *b , float *c , int n){
    extern __shared__ float sdata[];

    //2 shared mem arrays
    // s_a -> sdata[0]
    float *s_a = &sdata[0];
    // s_b -> sdata[256 (total no. of threads)]
    float *s_b = &sdata[blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    //loading data onto the shared mem
    if(i < n){
        s_a[tid] = a[i];
        s_b[tid] = b[i];
    }

    __syncthreads();

    if(i < n){
        c[i] = s_a[tid] + s_b[tid];
    }
}

void launchShared(float *a, float *b, float *c , int size){
    int blockSize = 256;
    int gridSize = (size + blockSize -1) / blockSize;

    vectorAdd_shared<<<gridSize , blockSize , 2*blockSize * sizeof(float)>>>(a, b, c ,size);
    cudaDeviceSynchronize();
}
