#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util.cuh"

#define HALF2(val) (*reinterpret_cast<half2*> (&(val)))

__global__ void vecadd_fp16x8_kernel(
    half* a,  half* b , half* c,
    int N
){

    // increase the work of single thread 
    // since half dtype has only x , y so we can do manual indexing -> 0 , 2 , 4 , 6
    // 0 will cover (0,1) , 2 -> (2 ,3) and so one till 6 -> (6,7)
    // so next index -> 8

    int global_index = (blockDim.x * blockIdx.x + threadIdx.x) * 8; // index with a stride of 8

    if(global_index < N){
        half2 reg_a1 = HALF2(a[global_index]);
        half2 reg_a2 = HALF2(a[global_index + 2]);
        half2 reg_a3 = HALF2(a[global_index + 4]);
        half2 reg_a4 = HALF2(a[global_index + 6]);

        half2 reg_b1 = HALF2(b[global_index + 0]);
        half2 reg_b2 = HALF2(b[global_index + 2]);
        half2 reg_b3 = HALF2(b[global_index + 4]);
        half2 reg_b4 = HALF2(b[global_index + 6]);

        half2 reg_c1 , reg_c2 , reg_c3 , reg_c4;

        reg_c1.x = __hadd(reg_a1.x , reg_b1.x);
        reg_c1.y = __hadd(reg_a1.y , reg_b1.y);
        reg_c2.x = __hadd(reg_a2.x , reg_b2.x);
        reg_c2.y = __hadd(reg_a2.y , reg_b2.y);
        reg_c3.x = __hadd(reg_a3.x , reg_b3.x);
        reg_c3.y = __hadd(reg_a3.y , reg_b3.y);
        reg_c4.x = __hadd(reg_a4.x , reg_b4.x);
        reg_c4.y = __hadd(reg_a4.y , reg_b4.y);


        if((global_index + 0) < N){
            HALF2(c[global_index + 0]) = reg_c1;
        }

        if((global_index + 2) < N){
            HALF2(c[global_index + 2]) = reg_c2;
        }

        if((global_index + 4) < N){
            HALF2(c[global_index + 4]) = reg_c3;
        }

        if((global_index + 6) < N){
            HALF2(c[global_index + 6]) = reg_c4;
        }
    }
}


void launch_fp16x8_kernel(half* a , half* b , half* c , int N){
    // each thread is processing 8 elements so total threads -> N / 8 
    int threads_needed = (N+7)/8;
    int block_size = 1024;
    int grid_size = (threads_needed + block_size -1) / block_size;
    vecadd_fp16x8_kernel<<<grid_size , block_size>>>(a, b, c, N);
    cudaDeviceSynchronize();
}


int main(){

    long long int N = 100000000;
    int iter = 100;

    vec_add::run_vectorAddBenchmark<half>(launch_fp16x8_kernel , N ,iter);

}