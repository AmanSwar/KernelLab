#include <cmath>
#include <cuda_runtime.h>
#include "../include/softmax_kernel.h"
__global__ void warp_optimized_softmax(float* input, float* output, int N, int D){

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;

    int warpsPerBlock = blockDim.x / 32;
    int warpId = threadIdx.x / 32;


    if(row < N){
        float thread_max = -INFINITY;

        for (int i = lane; i < D; i += 32) {
            thread_max = fmaxf(thread_max, input[row * D + i]);
        }

        #pragma unroll
        for(int offset = 16 ; offset > 0 ; offset /=2){
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }


        float max_val = __shfl_sync(0xffffffff, thread_max, 0);

        float thread_sum = 0.0f;
        for (int i = lane; i < D; i += 32) {
            float exp_val = expf(input[row * D + i] - max_val);
            output[row * D + i] = exp_val;
            thread_sum += exp_val;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        float sum = __shfl_sync(0xffffffff, thread_sum, 0);


        for (int i = lane; i < D; i += 32) {
            output[row * D + i] /= sum;
        }
    }


}


// Launch parameters example
void launch_warp_optimized_softmax(float* d_input, float* d_output, int N, int D) {
    // One warp per row
    int threadsPerBlock = 128; // 4 warps per block
    int blocksPerGrid = N;
    
    warp_optimized_softmax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, D);
}