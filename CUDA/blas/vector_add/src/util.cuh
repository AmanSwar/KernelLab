#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_ERROR(call) \
  do{\
    cudaError_t error = call; \
    if(error != cudaSuccess){ \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << "at line " << __LINE__ << std::endl \
      exit(EXIT_FAILURE); \
    }\
  } while(0)



namespace vec_add{

// unit test
#include <cstdlib>
inline bool test(float *kernel_out, float *ans, int N, float err = 1e-4) {
  for (int i = 0; i < N; i++) {
    if (std::abs(kernel_out[i] - ans[i]) > err) {
      return false;
    }
  }

  return true;
}

inline void init_arr(float* a , float* b , int N) {
    for(int i = 0 ; i < N ; i++){
        a[i]= i;
        b[i]= i;
    }
}


inline void cpu_vecAdd(float* a , float* b , float* c , int N){

  
  std::transform(
    a,
    a + N,
    b,
    c,
    std::plus<int>()
  );

}



void run_vectorAddBenchmark(void (*kernel)(float* ,  float* , float* , int) , int N ,int iter){

  float* a = new float[N];
  float* b = new float[N];
  float* c = new float[N];
  float* ko = new float[N]; // kernel output 

  init_arr(a , b , N);

  float* da , *db, *dc;
  cudaMalloc(&da, N * sizeof(float));
  cudaMalloc(&db, N * sizeof(float));
  cudaMalloc(&dc, N * sizeof(float));

  cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cpu_vecAdd(a , b , c,N);
  

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);


  float total_time = 0.0;

  for(int i = 0 ; i < iter ; i++){
    cudaEventRecord(start , 0);

    kernel(da , db , dc ,N);

    cudaEventRecord(end , 0);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);

    total_time += ms;

  }

  float total_ms = total_time/iter;
  std::cout << "Total Time : " << total_ms << std::endl;
  std::cout << "GFLOP : " << (N / (total_ms /1000)) / 1e+9 << std::endl;

  cudaMemcpy(ko , dc , N * sizeof(float) , cudaMemcpyDeviceToHost);

  std::cout << test(ko , c , N) << std::endl;
}
}