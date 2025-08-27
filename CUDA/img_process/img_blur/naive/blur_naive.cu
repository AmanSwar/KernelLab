#include "ATen/core/TensorBody.h"
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__
void blurKernel(float *input , float *output , int width , int height){
    // declaring thread index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // limit of x and y
    if( x >= width || y >= height) return;

    // sum and count for 1 iter
    float sum = 0.0f;
    int count = 0;

    // convolution img blue -> convoloving over 3x3 neighbour
    for(int dy = -1 ; dy <= 1 ; dy++){
        for(int dx = -1 ; dx <= 1 ; dx++){
            int nx = x + dx;
            int ny = y + dy;
            

            if(nx >= 0 && nx < width && ny >= 0 && ny < height){
                sum += input[ny * width + nx];
                count ++ ;
            }
        }
    }

    output[y * width + x] = sum/count;
}

// 
torch::Tensor blur(torch::Tensor input) {
    const int width = input.size(1);
    const int height = input.size(0);

    auto output = torch::zeros_like(input);

    // Launch CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    blurKernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        width, height
    );

    return output;
}



// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blur", &blur, "CUDA Image Blur");
}