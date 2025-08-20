#include "../include/softmax_kernel.h"
#include <cuda_runtime.h>

template <typename T, int THREADS_PER_BLOCK, int ITEMS_PER_THREAD>
__global__ void fused_softmax_kernel(T* input, T* output, int N, int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = THREADS_PER_BLOCK / 32;
    
    // Shared memory for reductions
    __shared__ float s_max[32];    // Max value per warp
    __shared__ float s_sum[32];    // Sum per warp
    
    if (row >= N) return;
    
    // Register cache for input elements
    float thread_data[ITEMS_PER_THREAD];
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    for (int base_col = 0; base_col < D; base_col += THREADS_PER_BLOCK * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int col = base_col + tid + i * THREADS_PER_BLOCK;
            if (col < D) {
                thread_data[i] = static_cast<float>(input[row * D + col]);
                thread_max = fmaxf(thread_max, thread_data[i]);
            } else {
                thread_data[i] = -INFINITY;
            }
        }
    }
    
    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // Store warp max to shared memory
    if (lane_id == 0) {
        s_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Block reduction for max using first warp only
    if (tid < 32) {
        float val_max = (tid < num_warps) ? s_max[tid] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_max = fmaxf(val_max, __shfl_down_sync(0xffffffff, val_max, offset));
        }
        if (tid == 0) {
            s_max[0] = val_max;
        }
    }
    __syncthreads();
    
    // Broadcast max to all threads
    const float max_val = s_max[0];
    
    // Compute exp(x - max) and local sum across all elements
    thread_sum = 0.0f;
    for (int base_col = 0; base_col < D; base_col += THREADS_PER_BLOCK * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int col = base_col + tid + i * THREADS_PER_BLOCK;
            if (col < D) {
                thread_data[i] = expf(static_cast<float>(input[row * D + col]) - max_val);
                thread_sum += thread_data[i];
            }
        }
    }
    
    // Warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Store warp sum to shared memory
    if (lane_id == 0) {
        s_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Block reduction for sum using first warp only
    if (tid < 32) {
        float val_sum = (tid < num_warps) ? s_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_sum += __shfl_down_sync(0xffffffff, val_sum, offset);
        }
        if (tid == 0) {
            s_sum[0] = val_sum;
        }
    }
    __syncthreads();
    
    // Broadcast sum to all threads
    const float sum = s_sum[0];
    
    // Write normalized values to output across all elements
    for (int base_col = 0; base_col < D; base_col += THREADS_PER_BLOCK * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int col = base_col + tid + i * THREADS_PER_BLOCK;
            if (col < D) {
                output[row * D + col] = static_cast<T>(expf(static_cast<float>(input[row * D + col]) - max_val) / sum);
            }
        }
    }
}

// Launch parameters with auto-tuning
void launch_fused_softmax(float* d_input, float* d_output, int N, int D) {
    const int threadsPerBlock = 128;
    int itemsPerThread;
    
    // Auto-tune items per thread based on D
    if (D <= 128) {
        itemsPerThread = 1;
    } else if (D <= 512) {
        itemsPerThread = 4;
    } else if (D <= 2048) {
        itemsPerThread = 8;
    } else {
        itemsPerThread = 16;
    }
    
    // Dynamically dispatch based on tuned parameters
    if (itemsPerThread == 1) {
        fused_softmax_kernel<float, threadsPerBlock, 1><<<N, threadsPerBlock>>>(d_input, d_output, N, D);
    } else if (itemsPerThread == 4) {
        fused_softmax_kernel<float, threadsPerBlock, 4><<<N, threadsPerBlock>>>(d_input, d_output, N, D);
    } else if (itemsPerThread == 8) {
        fused_softmax_kernel<float, threadsPerBlock, 8><<<N, threadsPerBlock>>>(d_input, d_output, N, D);
    } else {
        fused_softmax_kernel<float, threadsPerBlock, 16><<<N, threadsPerBlock>>>(d_input, d_output, N, D);
    }
}