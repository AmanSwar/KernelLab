#include <torch/extension.h>
#include "../include/greyscale_kernel.h"




torch::Tensor greyscale(torch::Tensor image){
    int batch_size , channels , height , width;

    batch_size = image.size(0);
    channels = image.size(1);
    height = image.size(2);
    width = image.size(3);
    auto output = torch::zeros_like(image);


    launch_naive(image.data_ptr<float>(), output.data_ptr<float>(), width, height, channels);

    return output;
}