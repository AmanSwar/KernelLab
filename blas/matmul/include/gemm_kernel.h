#ifndef GEMM_KERNEL_H
#define GEMM_KERNEL_H

void launch_gemm_naive(float *d_A , float *d_B , float *d_C , float alpha , float beta , int M , int N , int K);

void launch_gemm_tiled(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K);


void launch_gemm_optiled(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K);


void launch_gemm_regblock(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K);



void launch_gemm_warp(float *d_A, float *d_B, float *d_C, int M, int N, int K);



#endif