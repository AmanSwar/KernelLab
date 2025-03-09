#include <cuda_runtime.h>


__global__ void bitonic_sort(int* data, int j, int k, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int ixj = i ^ j;

    if (ixj > i && i < n && ixj < n){

        if ((i & k) == 0){

            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }

        }

        else{
            if (data[i] < data[ixj]){ 
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }

}   


void bitonicSort(int* data, int n){

    int power = 1;
    
    while(power < n) power *=2;
    
    dim3 blockSize(256);
    dim3 gridSize((power + blockSize.x - 1) / blockSize.x);

    for(int k = 2 ; k < power ; k *= 2){
        for(int j = k/2 ; j > 0 ; j /=2){
            bitonic_sort<<<gridSize , blockSize>>>(data, j, k, n);
            cudaDeviceSynchronize();
        }
    }
}