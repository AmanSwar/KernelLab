#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "util.cuh"


#define HALF2(val) (*reinterpret_cast<half2 *>(&(val)))
#define HALF2_PTR(ptr) (reinterpret_cast<__half2*>(ptr))

__global__ void sigmoid_fp16x2x8_kernel(__half *input, __half *output, int N) {
  int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 32;

  // half -> x and y so each index will cover 2 elements
  //  a single thread can take 2 x 16 -> 32 -> 32 x 2 -> 64 bytes

  // idx 0 -> 0,1 | 2 ->

  // we are going to use intrinsics for everything
  const half2 one = __float2half2_rn(1.0f);
  const half2 neg_one = __float2half2_rn(-1.0f);

#pragma unroll
  for (int i = 0; i < 32; i += 2) {
    // i -> 0 , 2 , 4
    const half2 reg_a = __ldcg(&HALF2(input[global_idx + i]));
    half2 reg_b;
    // 1 / (1 + exp(-x))
    half2 _neg_a = __hmul2(reg_a, neg_one);
    half2 _exp_negA = h2exp(_neg_a);
    half2 denom = __hadd2(one, _exp_negA);
    reg_b = __h2div(one, denom);
    __stcg(&HALF2(output[global_idx + i]), reg_b);
  }
}



// __global__ void sigmoid_fp16x2x8_revised_kernel(__half* input , __half* output , int N){
//   int global_idx = 
// }

void launch_sigmoid_kernel(__half *input, __half *output, int N) {
  int block_dim = 1024;
  int threads_needed = (N + 31)/ 32; // one thread takes care of 16 values
  int grid_dim = (threads_needed + block_dim - 1) / block_dim;
  sigmoid_fp16x2x8_kernel<<<grid_dim, block_dim>>>(input, output, N);
}


int main() {
  int N = 1000000;
  int iter = 100;
  sigmoid::runBenchmark<half>(launch_sigmoid_kernel, N, iter);
}