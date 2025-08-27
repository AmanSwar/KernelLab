#include "../include/gemm_kernel.h"
#include <torch/extension.h>

torch::Tensor matmul_kernel(torch::Tensor matA , torch::Tensor matB){
    TORCH_CHECK(matA.is_cuda() , "mat A is not in CUDA");
    TORCH_CHECK(matB.is_cuda() , "mat A is not in CUDA");
    

    auto matOut = torch::zeros_like(matA);
    int alpha = 1;
    int beta = 0;
    int M = matA.size(0);
    int N = matB.size(1);
    int K = matA.size(1);
    launch_gemm_regblock(matA.data_ptr<float>(), matB.data_ptr<float>(), matOut.data_ptr<float>(), alpha, beta, M, N, K);

    return matOut;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME , m){
    m.def("matmul_kernel" , &matmul_kernel , "matmul optimized kernel");
}