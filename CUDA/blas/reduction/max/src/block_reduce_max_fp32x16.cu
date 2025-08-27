#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/util.cuh"
#include <cfloat>

#define FLOAT4(val) (*reinterpret_cast<float4 *>(&val))

template <const int NUM_THREADS = 1024>
__device__ float block_max_fp32x16(float *arr, int N) {

    //declare shared mem
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    __shared__ float block_max_smem;

    
    //indexing
    const int num_elements_per_thread = 16;
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    int base_index = global_index * 16;

    for(int i = 0; i < num_elements_per_thread / 4 ; i *= 4){
        float4 value;
        
        int local_index = base_index + i;
        if (global_index * 4 < N) {
          value = FLOAT4(arr[local_index]);
        } else { // check of edge cases when the above condition fails i.e for
                 // elements at index  -1 , -2 , -3
          if (global_index * 3 < N) { // for element -3
            value.x = arr[global_index];
            value.y = arr[global_index + 1];
            value.z = arr[global_index + 2];
            value.w = -FLT_MAX;
          } else if (global_index * 2 < N) { // for -2
            value.x = arr[global_index];
            value.y = arr[global_index + 1];
            value.z = -FLT_MAX;
            value.w = -FLT_MAX;
          } else if (global_index < N) { // for -1
            value.x = arr[global_index];
            value.y = -FLT_MAX;
            value.z = -FLT_MAX;
            value.w = -FLT_MAX;
          }
        }
    }

}