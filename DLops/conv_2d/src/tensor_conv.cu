#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_fp16.h>
#include <mma.h>


#define FILTER_SIZE 3
#define TILE_WIDTH 16


using namespace nvcuda;

__global__ void conv_tensorCores(__half *input , __half *output , __half *filter , int width , int height){

    __shared__ __half tile[TILE_WIDTH + FILTER_SIZE -1][TILE_WIDTH + FILTER_SIZE -1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x  = blockIdx.x  * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    int filter_half = FILTER_SIZE /2;

    int shared_x = tx + filter_half;
    int shared_y = ty +filter_half;

    tile[shared_y][shared_x] = (x < width && y < height) ? __ldg(&input[y * width + x]) : __float2half(0.0f);

    __syncthreads();

    __half sum = __float2half(0.0f);

    wmma::fragment<wmma::matrix_a , 4 ,4 ,4 , __half , wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b , 4 , 4, 4 , __half , wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator , 4 , 4 , 4 , __half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    if(tx < TILE_WIDTH && ty < TILE_WIDTH && x < width && y < height){
        wmma::load_matrix_sync(a_frag , &tile[shared_y][shared_x] , TILE_WIDTH);
        wmma::load_matrix_sync(b_frag, filter, FILTER_SIZE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(&output[y * width + x], c_frag, width, wmma::mem_row_major);
    }

}

void launch_tensor_conv(__half *input , __half *output , __half *filter , int width , int height){
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_tensorCores<<<gridSize, blockSize>>>(input, output, filter, width, height);
    cudaDeviceSynchronize();
}