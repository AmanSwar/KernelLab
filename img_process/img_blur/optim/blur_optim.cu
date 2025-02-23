#include "ATen/core/TensorBody.h"
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__
void blurKernelShared(
    float* input , 
    float* output,
    int width,
    int height
){

    __shared__ float tile[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    // load data into shared mem
    if(x < width && y < height){
        tile[ty + 1][tx + 1] = input[y * width + x];
    }

    // loading halo pixels
    if(tx == 0 && x > 0){
        tile[ty + 1][0] = input[y * width + (x - 1)];
    }

    if(tx == 15 && x < width -1){
        tile[ty +1][17] = input[y * width + (x + 1)];

    }

    if(ty == 0 && y > 0){
        tile[0][tx + 1] = input[(y-1) * width + x];
    }

    if(ty == 15 && y < height-1){
        tile[17][tx+1] = input[(y + 1) * width + x];
    }

    __syncthreads();    

    if ( x < width && y < height){
        float sum = 0.0f;
        for(int i = -1 ; i <= 1; i++){
            for(int j = -1 ; j <=1 ;j++){
                sum += tile[ty + 1 + i][tx + 1 + j];
            }
        }

        output[y * width + x] = sum / 9.0;
    }


}

torch::Tensor blur(torch::Tensor input){
    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::zeros_like(input);

    dim3 blockSize(16 , 16);
    dim3 gridSize((width + 15)/16 , (height  + 15) / 16);

    blurKernelShared<<<gridSize , blockSize>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        width,
        height
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blur", &blur, "CUDA Image Blur");
}