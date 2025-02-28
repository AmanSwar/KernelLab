
#include <cuda_runtime.h>

#include "../include/vec_add_kernel.h"
#define TILE_SIZE 128


__global__
void tiled_vec_add(float*a , float*b , float*c , int n){
    

    int tile_start = blockIdx.x * TILE_SIZE;

    for(int i = threadIdx.x; i < TILE_SIZE && tile_start + i < n ; i += blockDim.x){
        c[tile_start + i] = a[tile_start + i] + b[tile_start + i];
    }
}

void launchTiled(float *a, float *b, float *c, int size){

    int blockSize = 256;
    int gridSize = (size + TILE_SIZE -1) / TILE_SIZE;

    tiled_vec_add<<<gridSize , blockSize>>>(a  , b , c ,size);
    cudaDeviceSynchronize();

}



