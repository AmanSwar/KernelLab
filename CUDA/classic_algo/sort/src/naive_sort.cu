#include <cuda_runtime.h>

/*
implementation of parallel bubble sort
*/

__global__ void naive_sort(int* data, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i= 0 ; i < n ; i++){

        //odd phase
        if(i % 2 == 1){

            if(idx * 2 + 1 < n-1){
                if(data[idx * 2 + 1] > data[idx * 2 + 2]){
                    //swap
                    int temp = data[idx * 2 + 1];
                    data[idx * 2 + 1] = data[idx * 2 + 2];
                    data[idx * 2 + 2] = temp;
                }
            }

        }
        //even
        else{

            if(idx * 2 < n-1){
                if(data[idx * 2] > data[idx * 2 +1]){
                    int temp = data[idx * 2];
                    data[idx * 2] = data[idx * 2 + 1];
                    data[idx * 2 + 1] = temp;
                }
            }

        }

        __syncthreads();
    }


}


void launch_naive_sort(int *data , int n){
    int blockSize = 256;
    int gridSize = (n/2 + blockSize -1) / blockSize;

    naive_sort<<<gridSize , blockSize>>>(data , n);
    cudaDeviceSynchronize();
}