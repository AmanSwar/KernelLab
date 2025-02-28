#include "../include/vec_add_kernel.h"

#include <torch/extension.h>

torch::Tensor naive_vec_add(torch::Tensor vecA , torch::Tensor vecB){
    TORCH_CHECK(vecA.is_cuda() , "vecA not in CUDA");
    TORCH_CHECK(vecB.is_cuda() , "vecB not in CUDA");
    int size = vecA.size(0);

    auto output = torch::zeros_like(vecA);
    launchVecAdd(vecA.data_ptr<float>(), vecB.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME , m){
    m.def("naive_vec_add" , &naive_vec_add , "naive implementation of vector addition");
}

