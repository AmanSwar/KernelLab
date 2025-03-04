#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__
void transShared(float *A , float *B , int N){
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //load into shared mem
    if(y < N && x < N){
        // map in the same way without transpose
        tile[ty][tx] = A[y * N + x];

    }

    __syncthreads();


    // indices for transposed mat => y -> x && x -> y
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if(y_out < N && x_out < N){
        //transpose
        B[x_out * N + y_out] = tile[ty][tx];
    }

}


void launch_shared(float *A , float *B , int N){
    dim3 blockSize(BLOCKSIZE , BLOCKSIZE);
    dim3 gridSize((N + BLOCKSIZE -1)/ BLOCKSIZE ,(N + BLOCKSIZE -1)/ BLOCKSIZE);
    transShared<<<gridSize , blockSize>>>(A, B, N);
    cudaDeviceSynchronize();
}