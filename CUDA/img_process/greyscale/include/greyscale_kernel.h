
#ifndef GREYSCALE_KERNEL_H
#define GREYSCALE_KERNEL_H


void launch_naive(const float *input , float *output , int width , int height , int channels);

void launch_coal(const float *input , float *output , int width , int height , int channels);

void launch_multipixel(const float *input , float *output , int width , int height , int channels , int pixelsPerThread);

void launch_fma(const float *input , float *output , int width , int height);

void launch_shared(const float *input , float *output , int width , int height , int channels);

void launch_vectorized(const float4 *input , float *output , int width , int height);



#endif