#include <cuda_runtime.h>

__global__ void gemm_optim(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K) {


    const int BM = 64;  
    const int BN = 64;  
    const int BK = 16;  
    
  
    const int TM = 8;   
    const int TN = 8;   


    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

 

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    

    const int row_start = by * BM + ty * TM;
    const int col_start = bx * BN + tx * TN;

   
    float reg_C[TM][TN] = {0.0f};

 
    const int thread_block_dim_x = 8;  // BN/TN = 64/8 = 8
    const int thread_block_dim_y = 8;  // BM/TM = 64/8 = 8


    for (int t = 0; t < (K + BK - 1) / BK; t++) {
      
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            for (int k = 0; k < BK; k += thread_block_dim_x) {
                int m_idx = row_start + m;
                int k_idx = t * BK + k + tx;
                
                if (m_idx < M && k_idx < K) {
                    s_A[ty * TM + m][k + tx] = A[m_idx * K + k_idx];
                } else {
                    s_A[ty * TM + m][k + tx] = 0.0f;
                }
            }
        }
        


        #pragma unroll
        for (int k = 0; k < BK; k += thread_block_dim_y) {
            for (int n = 0; n < TN; n++) {
                int k_idx = t * BK + k + ty;
                int n_idx = col_start + n;
                
                if (k_idx < K && n_idx < N) {
                    s_B[k + ty][tx * TN + n] = B[k_idx * N + n_idx];
                } else {
                    s_B[k + ty][tx * TN + n] = 0.0f;
                }
            }
        }
        
        __syncthreads();


        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    reg_C[m][n] += s_A[ty * TM + m][k] * s_B[k][tx * TN + n];
                }
            }
        }
        
        __syncthreads();
    }




    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int r = row_start + m;
            int c = col_start + n;
            
            if (r < M && c < N) {
                C[r * N + c] = alpha * reg_C[m][n] + beta * C[r * N + c];
            }
        }
    }
}

void launch_gemm_optiled(float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K) {
 
    const int thread_block_dim_x = 8;
    const int thread_block_dim_y = 8;

    const int BM = 64;
    const int BN = 64;
    
    dim3 threadsPerBlock(thread_block_dim_x, thread_block_dim_y);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_optim<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}