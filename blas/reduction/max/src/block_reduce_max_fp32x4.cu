#include <cfloat>   
#include <cuda_runtime.h>
#include "../include/util.cuh"

#define FLOAT4(val) (*reinterpret_cast<float4*>(&val))

//return the block max val
template <const int NUM_THREADS = 1024>
__device__ float block_max_fp32x4(float *arr , int N){
    //shared mem for storing local maximum of each warps
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float warp_max_smem[NUM_WARPS];
    __shared__ float block_max_smem;

    
    int global_index = (blockDim.x * blockIdx.x + threadIdx.x);
    //float -> float4 conversion , hence each base index will take its neighbour 4 values and make it float4
    int base_index = global_index * 4;

    int local_index = threadIdx.x;
    //register value -> hold all the value from global mem
    float4 value;

    if(global_index * 4 < N){
        value = FLOAT4(arr[base_index]);
    }else{ // check of edge cases when the above condition fails i.e for elements at index  -1 , -2 , -3
        if(global_index * 3 < N){ // for element -3
            value.x = arr[global_index];
            value.y = arr[global_index + 1];
            value.z = arr[global_index + 2];
            value.w = -FLT_MAX;
        }
        else if(global_index * 2 < N){ // for -2
            value.x = arr[global_index];
            value.y = arr[global_index + 1];
            value.z = -FLT_MAX;
            value.w = -FLT_MAX;
        }
        else if(global_index < N){ // for -1
            value.x = arr[global_index];
            value.y = -FLT_MAX;
            value.z = -FLT_MAX;
            value.w = -FLT_MAX;
        }


    }

    // find local maxima withing each float4
    float _sub_max1 = max(value.x , value.y);
    float _sub_max2 = max(value.z , value.w);
    float _max = max(_sub_max1 , _sub_max2);


    float max_per_warp = warp_reduce_max(_max); // each thread has a register with local maximum

    int warp_id = local_index / WARP_SIZE;
    int lane_id = local_index % WARP_SIZE;

    // for 1st thread/lane in each warp
    if(lane_id == 0){
        warp_max_smem[warp_id] = max_per_warp;
    }
    __syncthreads();

    //now load from shared mem to get all local maximum
    _max = (local_index < (NUM_WARPS)) ? warp_max_smem[local_index] : -FLT_MAX;

    //for the first warp
    if(warp_id == 0){
        //each thread in warp will hold value
        float block_max = warp_reduce_max(_max);
        
        //for first lane
        if(lane_id == 0){
            block_max_smem = block_max;
        }
    
    }
    __syncthreads();

    return block_max_smem;
}


__global__ void reduce_pass1(float* a , float* b , int N){
    /*
    First pass -> stores all the block max to global memory
    */    

    float block_max = block_max_fp32x4(a, N);

    if(threadIdx.x == 0){
        b[blockIdx.x]  = block_max;
    }
}


__global__ void reduce_pass2(float* b , int N){
    /*
    Second pass to find maximum among the max(block)
    */
    float final_max = block_max_fp32x4(b,  N);

    if(threadIdx.x == 0){
        b[0] = final_max;
    }
}


void launch_block_reduce(float* a , float* b, int N){
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;

    reduce_pass1<<<grid_dim , block_dim>>>(a, b, N);

    if(grid_dim > 1){
        reduce_pass2<<<1 , block_dim>>>(b, N);
    }
}

int main() {
  int N = 100000;
  int iter = 100;

  reduceBenchmark(launch_block_reduce, N, iter);
}
