#ifndef REDUCTION_KERNELS
#define REDUCTION_KERNELS


void launch_naive_reduction(float* d_input, float* d_output, int N);
void launch_no_divergence_reduction(float* d_input, float* d_output, int N);
void launch_warp_optimized_reduction(float* d_input, float* d_output, int N);


#endif