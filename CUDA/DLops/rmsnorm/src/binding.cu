#include "ATen/core/TensorBody.h"
#include "c10/util/BFloat16.h"

#include "../src/naive_rms.cuh"

#include <cassert>
#include <torch/extension.h>


extern "C" void launch_cuda_rms_bf16(
    const nv_bfloat16* input_matrix,
    const nv_bfloat16* weight_matrix,
    nv_bfloat16* out_matrix,
    int M,
    int N,
    float eps
){
  launch_rms_bf16(input_matrix, weight_matrix, out_matrix, M, N, eps);
}



torch::Tensor fused_rmsnorm(
    torch::Tensor input_matrix,
    torch::Tensor weight,
    float eps = 1e-6f
) {
    TORCH_CHECK(input_matrix.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input_matrix.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(input_matrix.dim() == 3, "expected input shape (bs, seq_len, embed_dim)");

    // ensure contiguous
    if (!input_matrix.is_contiguous()) input_matrix = input_matrix.contiguous();
    if (!weight.is_contiguous()) weight = weight.contiguous();

    int64_t bs = input_matrix.size(0);
    int64_t seq_len = input_matrix.size(1);
    int64_t embed_dim = input_matrix.size(2);

    int M = static_cast<int>(bs * seq_len); // rows
    int N = static_cast<int>(embed_dim);    // columns

    TORCH_CHECK(weight.size(0) == N, "weight size must match embed dim");

    auto output = torch::empty_like(input_matrix);

    const nv_bfloat16* in_ptr = reinterpret_cast<const nv_bfloat16*>(input_matrix.data_ptr<at::BFloat16>());
    const nv_bfloat16* w_ptr = reinterpret_cast<const nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
    nv_bfloat16* out_ptr = reinterpret_cast<nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    launch_cuda_rms_bf16(in_ptr, w_ptr, out_ptr, M, N, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_kernel", &fused_rmsnorm, "Fused RMSNorm (BF16)");
}