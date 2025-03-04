#include <cuda_runtime.h>
#include "../include/gemm_kernel.h"


__global__ void gemm_optim(float *A , float *B , float *C, float alpha , float beta , int M , int N , int K){

    // constants
    const int BM = 64;
    const int BN = 64;
    const int BK = 64;
    const int TM = 4; 
    const int TN = 4; 

    //shared memory
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];


    // indexes
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;



    float reg_C[TM][TN] = {0.0f};  //register


    // number of tiles 
    for(int t = 0 ; t < (K + BK - 1) / BK ; t++){

        #pragma unroll
        for(int i = 0; i < BM; i += blockDim.y) {
            if(ty + i < BM && tx < BK && t * BK + tx < K) {
                s_A[ty + i][tx] = A[(by * BM + ty + i) * K + t * BK + tx];
            } else if(ty + i < BM && tx < BK) {
                s_A[ty + i][tx] = 0.0f;
            }
        }

        #pragma unroll
        for(int i = 0; i < BK; i += blockDim.y) {
            if(ty + i < BK && t * BK + ty + i < K && tx < BN) {
                s_B[ty + i][tx] = B[(t * BK + ty + i) * N + bx * BN + tx];
            } else if(ty + i < BK) {
                s_B[ty + i][tx] = 0.0f;
            }
        }

        __syncthreads();


        #pragma unroll
        for(int k = 0 ; k < BK ; k++){
            #pragma unroll
            for(int m = 0 ; m < TM ; m++){
                #pragma unroll
                for(int n = 0 ; n < TN ; n++){
                    reg_C[m][n] += s_A[ty * TM + m][k] * s_B[k][tx * TN + n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int m = 0 ; m < TM ; m++){
        for(int n = 0 ; n < TN ; n++){
            int r = row + m;
            int c = col + n;
            if(r < M && c < N){
                C[r * N + c] = alpha * reg_C[m][n] + beta * C[r * N + c];
            }
        }
    }
}


void launch_gemm_optiled(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K){
    dim3 threadsPerBlock(16 , 16);
    dim3 blocksPerGrid((N + 128 - 1)/ 128  , (M + 128 - 1) / 128);

    gemm_optim<<<blocksPerGrid , threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta,M, N, K);
    cudaDeviceSynchronize();
}   