#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


__global__
void radix_sort(int* data , int* tempData , int n , int bit){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        bool isSet = (data[i] >> bit) & 1; 

        int blockCount = __syncthreads_count(isSet);

        __shared__ int blockOffset , falseOffset;


        if(threadIdx.x == 0){

            if(blockCount > 0){
                atomicAdd(&blockOffset , blockCount);

            }

            falseOffset = blockIdx.x * blockDim.x - blockOffset;
        }
        __syncthreads();

        int position;

        if(isSet){
            position = blockOffset + threadIdx.x;

        }

        else{
            position = falseOffset +  threadIdx.x;
        }


        if(i < n){
            tempData[position] = data[i];
        }
    }
}



void launch_naive_sort(int *data , int* tempData , int n , int bit){
    int blockSize = 256;
    int gridSize = (n + blockSize -1) / blockSize;
    radix_sort<<<gridSize , blockSize>>>(data, tempData, n, bit);
    cudaDeviceSynchronize();
}