#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

void launch_naive_conv(float *input , float *output , float *filter , int width , int height , int filter_size);

void launch_tiled_conv(float *input , float *output , float *filter , int width , int height);

void launch_coal_conv(float *input , float *output , float *filter , int width , int height);

void launch_tensor_conv(__half *input , __half *output , __half *filter , int width , int height);

#endif 