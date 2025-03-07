#ifndef CONV_3D_KERNEL_H
#define CONV_3D_KERNEL_H

void launch_conv3d_naive(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
);

void launch_conv3d_shared(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
);

void launch_conv3d_tiled(
    float* input , float *kernel , float *output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co, int Dk, int Hk, int Wk,
    int Do, int Ho, int Wo,
    int TILE_D , int TILE_H , int TILE_W
);


void launch_conv3d_optim(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co , int Dk , int Hk , int Wk,
    int Do, int Ho, int Wo,
    int TILE_H , int TILE_D , int TILE_W , int VEC_SIZE,
);





#endif