#include <asm-generic/errno.h>
#include <climits>
#include <cuda_runtime.h>

__global__ void optim_bitonic_sort(int* data, int n){
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = data[i];
    }
    else{
        //padding
        sdata[tid] = INT_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            unsigned int ixj = tid ^ j;
            
            if (ixj > tid) {

                if ((tid & k) == 0) {


                    if (sdata[tid] > sdata[ixj]) {
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                } 
                else {
                    if (sdata[tid] < sdata[ixj]) {
                  
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (i < n) {
        data[i] = sdata[tid];
    }


}

void optim_bitonicSort(int* data , int n){
    int power = 1;
    
    while(power < n) power *=2;
    dim3 blockSize(1024);
    dim3 gridSize((power + blockSize.x - 1) / blockSize.x);

    optim_bitonic_sort<<<gridSize , blockSize>>>(data, n);
    cudaDeviceSynchronize();

}