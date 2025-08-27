#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

void launch_naive_softmax(float* d_input, float* d_output, int N, int D);

void launch_shared_memory_softmax(float* d_input, float* d_output, int N, int D);

void launch_warp_optimized_softmax(float* d_input, float* d_output, int N, int D);

void launch_block_optimized_softmax(float* d_input, float* d_output, int N, int D);


void launch_fused_softmax(float* d_input, float* d_output, int N, int D);


#endif
