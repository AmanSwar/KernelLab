#pragma once
#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <iostream>
#include <cfloat>

#define FLOAT4(val) (*reinterpret_cast<float4 *>(&val))

#define WARP_SIZE 32

template <const int warp_size = WARP_SIZE>
__device__ float warp_reduce_max(float val){
    unsigned int mask = 0xFFFFFFFF;

    for(int offset = WARP_SIZE >> 1 ; offset >=1 ; offset >>=1){
        val = max(val , __shfl_xor_sync(mask , val , offset));
    }

    return val;
}

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





inline bool softmaxCUDNN(float* input , float* output , int iter){
    const int BATCH_SIZE = 1;
    const int C = 1;
    const int H = 1;
    const int W = 1;

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t input_desc , output_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        BATCH_SIZE,
        C, H , W        
    );

    cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        BATCH_SIZE, C , H , W
    );


    float alpha =  1.0f;
    float beta = 0.0f;
   cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        input_desc,
        input,
        &beta,
        output_desc,
        output   
    );

    cudaEvent_t start , end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i = 0 ; i < iter ; i++){
        cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            input_desc,
            input,
            &beta,
            output_desc,
            output   
        );
        
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms , start , end);


    std::cout << "CUDNN TIME : " << ms << std::endl;

}


template <class dtype>
inline bool verify(
    dtype* kernel_out,
    dtype* cudnn_out,
    int N,
    float error = 1e-4
){
    for(int i = 0 ; i < N ; i++){
        if(kernel_out[i] - cudnn_out[i] > error){
            return false;
        }
    }
    return true;
}

template <class dtype>
inline void init(dtype* a , int N){
    
    for(int i = 0 ; i < N ; i++){
        a[i] = rand();
    }

}

template <class dtype>
void benchmarkSoftmax(void (*kernel)(dtype* input , dtype* output , int) , int N , int iter){
    

    dtype* a = new dtype[N];
    dtype* b = new dtype[N];

    init(a);

    dtype* input , *output , *cudnn_output;

    cudaMalloc(&input , sizeof(dtype) * N);
    cudaMalloc(&output , sizeof(dtype) * N);
    cudaMalloc(&cudnn_output , sizeof(dtype)*N);

    cudaMemcpy(input , a , sizeof(dtype) * N , cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start , end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //warmup run
    kernel(input , output , N);


    cudaEventRecord(start);

    for(int i = 0 ; i < iter ; i++){
        kernel(input , output , N);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);

    std::cout << "KERNEL TIME : " << ms << std::endl;
    // std::cout << "KERNEL GFLOPS : " << 

    
}

