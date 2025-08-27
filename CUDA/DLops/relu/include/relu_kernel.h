#ifndef RELU_KERNEL_H
#define RELU_KERNEL_H
void launch_relu_naive(float* d_input, float* d_output, int size);
void launch_relu_vectorized(float * input , float * output , int size);
void launch_relu_optimized(float* d_input, float* d_output, int size);
void launch_relu_fp32x16(float *d_input, float *d_output, int size);

#endif 