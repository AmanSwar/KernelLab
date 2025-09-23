#pragma once

#include <chrono>
#include <iostream>

#include <torch/torch.h>

void torch_transpose(float* x , float* y , int M , int N){
  torch::Tensor input_tensor = torch::from_blob(x, {M, N}, torch::kFloat32);

  auto start = std::chrono::high_resolution_clock::now();
  torch::Tensor output_tensor;
  for(int i = 0 ; i < 100 ; i++){
    output_tensor = input_tensor.transpose(0, 1);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  torch::Tensor contiguous_output_tensor = output_tensor.contiguous();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  float ms = duration.count();
  std::cout << "Torch time : " << ms/100 << " microseconds" << std::endl;
  std::cout << "Torch GFLOPS : " << ((N*N) /  ((ms/100)/1000)) * 1e-9 << std::endl;
}

