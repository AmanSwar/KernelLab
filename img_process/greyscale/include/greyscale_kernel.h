#ifndef GREYSCALE_KERNEL_H
#define GREYSCALE_KERNEL_H

void launch_naive(const float *input , float *output , int width , int height , int channels);

void launch_coal(const float *input , float *output , int width , int height , int channels);
#endif