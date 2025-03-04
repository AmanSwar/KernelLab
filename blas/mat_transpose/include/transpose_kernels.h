#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

void launch_naive(float *A , float *B , int N);

void launch_shared(float *A , float *B , int N);



#endif