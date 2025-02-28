#include <cuda_runtime.h>


__global__
void multiElement_vecadd(float *a , float *b , float *c , int n){
    const int ELEMENTS_PER_THREAD = 4;

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i + 3*blockDim.x < n){
        c[i] = a[i]  + b[i];
        c[i + blockDim.x] = a[i + blockDim.x] + b[i + blockDim.x];
        c[i + 2*blockDim.x] = a[i + 2*blockDim.x] + b[i + 2*blockDim.x];
        c[i + 3*blockDim.x] = a[i + 3*blockDim.x] + b[i + 3*blockDim.x];
    }
    else{
        for(int j = 0 ; j < ELEMENTS_PER_THREAD ; j++){
            int idx = i + j*blockDim.x;
            if(idx < n){
                c[idx] = a[idx] + b[idx];
            }
        }
    }
}


void launchMultiElement(float *a , float *b , float *c , int size){
    int blockSize = 256;
    int gridSize = (size + blockSize*4-1)/(blockSize * 4);
    multiElement_vecadd<<<gridSize , blockSize>>>(a,b,c,size);


}