
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cstddef>

#define WARP_SIZE 32

__device__ __forceinline__ half warp_reduce_sum_fp16(half val) {
  const uint mask = 0xffffffff;
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_xor_sync(mask, val, offset);
  }

  return val;
}


__global__ void matrix_reduce_sum_fp16_kernel(
    half* input_matrix, // input matrix of dimension M x N
    half* output_arr, // output array of dimension 1 x M consisting of sum of each row
    int M , int N 
){

    int local_index_x = threadIdx.x; // range -> 0 - blockDim.x
    
    if(blockIdx.x >= M) return;


    extern __shared__ half smem[]; // shared memory declaration
    

    const int total_warps = blockDim.x / WARP_SIZE;
    const int total_tiles = (N + blockDim.x - 1) / blockDim.x;
    

    half* smem_local_sum = &smem[0]; // contains per warp sum (size = total_warps)
    half* smem_per_tile_sum = &smem[total_warps]; // contains sum in a tile (size = total_tiles)

    int row_start = blockIdx.x * N;

    

        
    for(int tile = 0 ; tile < total_tiles ; tile ++){
        // idx range -> 0 - blockDim.x
        int col_idx = tile * blockDim.x + local_index_x;

        int warp_id = local_index_x /WARP_SIZE;
        int lane_id = local_index_x % WARP_SIZE;
        
        half val = (col_idx < N) ? input_matrix[row_start + col_idx] : __float2half(0.0f);
        
        val = warp_reduce_sum_fp16(val); // get the warp sum
        //store only the 0th lane of every warp
        if(lane_id == 0){
            smem_local_sum[warp_id] = val;
        }
        __syncthreads();
        

        //again load cuz find sum among warp
        val = (local_index_x < total_warps) ? smem_local_sum[local_index_x] : __float2half(0.0f);
        if(warp_id == 0){
            val = warp_reduce_sum_fp16(val);
        }

        if(local_index_x == 0){
            smem_per_tile_sum[tile] = val;
        }
        __syncthreads();
        
    }
    
    float sum = 0.0f; // acc should be in float
    
    if(local_index_x == 0){
        #pragma unroll
        for(int i = 0 ; i < total_tiles ; i++){
            sum += __half2float(smem_per_tile_sum[i]);
        }
        // __syncthreads();
    }

    if(local_index_x == 0){
        output_arr[blockIdx.x] = __float2half(sum);
    }


    

}

void launch_matrix_reduce_sum(
    half* input_matrix,
    half* output_arr,
    int M , int N
){

    
    int block_dim = 1024;
    int grid_dim = M; // total blocks = M cuz each block is handeling 1 row 

    const int total_warps_in_block = block_dim / WARP_SIZE;
    const int total_tiles = (N + block_dim - 1) / block_dim;
    const size_t smem_size = sizeof(half) * (total_warps_in_block + total_tiles);

    matrix_reduce_sum_fp16_kernel<<<grid_dim , block_dim , smem_size>>>(input_matrix , output_arr , M , N);
 
}




int main(){
    const int M = 1024;
    const int N = 4096;
    const int iter = 100; // number of iteration for loop

    half* hm  = new half[M*N];
    half* harr = new half[M]; // array to store the sum of all array
    half* hans = new half[M];

    //init
    for(int i = 0 ; i < M * N ; i++){
        hm[i] = __float2half(float(i) / (float)(M*N));
    }

    //allocate mem to device
    half* dm , *darr;
    cudaMalloc(&dm , sizeof(half) * M * N);
    cudaMalloc(&darr , sizeof(half) * M);

    cudaMemcpy(dm , hm , sizeof(half) * M * N , cudaMemcpyHostToDevice);

    //benchmarking
    cudaEvent_t start , end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    launch_matrix_reduce_sum(dm, darr, M, N);

    cudaEventRecord(start);
    for(int i = 0 ; i < iter ; i++){
        launch_matrix_reduce_sum(dm , darr , M ,N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "Kernel Time : " << ms / iter << std::endl;
    std::cout << "Kernel GFLOP : " << (double)(M * (N- 1) / (ms / iter/ 1000)) / 1e9 << std::endl;

    //copy  back the memory
    cudaMemcpy(harr, darr, sizeof(half) * M, cudaMemcpyDeviceToHost);

    //cpu version
    for (int i = 0; i < M; i++) {
      float sum = 0.0f; // Use float for CPU accumulator for better precision
      for (int j = 0; j < N; j++) {
        sum += __half2float(hm[i * N + j]);
      }
      hans[i] = __float2half(sum);
    }

    //verify
    for(int i = 0 ; i < M ; i++){
        if(std::abs(__half2float(harr[i]) - __half2float(hans[i])) > 1e-3){
            std::cout << static_cast<float>(harr[i]) << " " << static_cast<float>(hans[i]) << std::endl;
            std::cout << static_cast<float>(hm[i]) << std::endl;
            std::cout << "Fail";
            return 0;
        }
    }


    std::cout << "Pass" << std::endl;
    delete[] hm;
    delete[] harr;
    delete[] hans;
    cudaFree(dm);
    cudaFree(darr);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}