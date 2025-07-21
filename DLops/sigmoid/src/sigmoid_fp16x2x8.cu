#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <exception>

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



__global__ void sigmoid_fp16x2x8_revised_kernel(__half* input , __half* output , int N){
  const int elements_per_thread = 16; // reduce register pr.
  
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;  // 0 , el , el*2 ...
 
  //check boudns
  if(idx >= N){
    return;
  }

  
  const half2 one = __float2half2_rn(1.0f);

  //pointer of type half2 to get input as half2 : on fly conversion is expensive as done above
  half2* input_vec = reinterpret_cast<half2*>(input);
  half2* output_vec = reinterpret_cast<half2*>(output);

  int vec_idx = idx / 2;
  //total vectors to be loaded (vec size = 2)
  const int vec_per_thread = elements_per_thread / 2; 

  #pragma unroll
  for(int i = 0; i < vec_per_thread ; i++){
    int curr_vec_idx = vec_idx + i;

    if(curr_vec_idx * 2 < N){
      half2 reg_a = __ldcg(&input_vec[curr_vec_idx]);

      half2 _neg_a = __hneg2(reg_a); // negation instead of mul with -1
      half2 _exp_neg_a = h2exp(_neg_a); 
      half2 denom = __hadd2(one , _exp_neg_a);
      half2 reg_b = __h2div(one , denom);

      __stcg(&output_vec[curr_vec_idx] , reg_b);
    }
  }
}

void launch_sigmoid_kernel(__half *input, __half *output, int N) {
  int block_dim = 1024;
  int threads_needed = (N + 31)/ 32; // one thread takes care of 16 values
  int grid_dim = (threads_needed + block_dim - 1) / block_dim;
  sigmoid_fp16x2x8_kernel<<<grid_dim, block_dim>>>(input, output, N);
}

void launch_revised_kernel(__half* input , __half* output , int N){
  const int el_per_thread = 16;
  const int block_dim = 256;
  int threads_needed = (N + el_per_thread - 1) / el_per_thread;
  int grid_dim = (threads_needed + block_dim -1) / block_dim;
  const int num_sm = 16;
  int grid_dim_padded = (grid_dim + num_sm - 1) / num_sm * num_sm;
  sigmoid_fp16x2x8_revised_kernel<<<grid_dim_padded, block_dim>>>(input, output,
                                                                  N);
}


int main() {
  long int N = 1000000;
  int iter = 100;
  sigmoid::runBenchmark<half>(launch_revised_kernel, N, iter);
}