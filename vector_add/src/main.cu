#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include "../include/vec_add_kernel.h"

int main() {
    const int size = 10000000;
    float *a = new float[size];
    float *b = new float[size];
    float *c = new float[size];

    for(int i = 0; i < size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    float *da, *db, *dc;
    int size_allocated = size * sizeof(float);
    cudaMalloc(&da, size_allocated);
    cudaMalloc(&db, size_allocated);
    cudaMalloc(&dc, size_allocated);

    cudaMemcpy(da, a, size_allocated, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_allocated, cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    //for naive implementation
    cudaEventRecord(start);
    launchVecAdd(da, db, dc, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    printf("CUDA kernel execution time (naive) : %.6f s\n", naive_time/1000);
    

    //for shared memory 
    cudaEventRecord(start);
    launchShared(da, db, dc, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("CUDA kernel execution time (shared) : %.6f s\n", shared_time/1000);
    


    
    //for coalesced memory
    cudaEventRecord(start);
    launchCoalesced(da, db, dc, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float coalesced_time = 0;
    cudaEventElapsedTime(&coalesced_time, start, stop);
    printf("CUDA kernel execution time (coalesced) : %.6f s\n", coalesced_time/1000);
    

    //tiling
    cudaEventRecord(start);
    launchTiled(da , db ,dc , size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);
    printf("CUDA kernel execution time (tiled) : %.6f s\n", tiled_time/1000);
    

    //multi element processing
    cudaEventRecord(start);
    launchMultiElement(da, db , dc , size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float multi_time = 0;
    cudaEventElapsedTime(&multi_time, start, stop);
    printf("CUDA kernel execution time (multi) : %.6f s\n", multi_time/1000);


    cudaEventRecord(start);
    launchVectorized(da, db , dc , size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float vectorized_time = 0;
    cudaEventElapsedTime(&vectorized_time, start, stop);
    printf("CUDA kernel execution time (vectorized) : %.6f s\n", vectorized_time/1000);

    
    printf("\n");
    printf("Boost from shared : %2f \n" , naive_time/shared_time );
    printf("Boost from coalescing: %2f \n" , naive_time/coalesced_time );
    printf("Boost from tiled: %2f \n" , naive_time/tiled_time );
    printf("Boost from multi: %2f \n" , naive_time/multi_time );
    printf("Boost from vectorization: %2f \n" , naive_time/vectorized_time );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(c, dc, size_allocated, cudaMemcpyDeviceToHost);

    
    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    
    return 0;
}