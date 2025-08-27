#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

__global__ void conv3d_tiled(
    float* input , float *kernel , float *output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co, int Dk, int Hk, int Wk,
    int Do, int Ho, int Wo,
    int TILE_D , int TILE_H , int TILE_W
){
    extern __shared__ float shared_mem[];

    float * shared_input = shared_mem;
    float *shared_kernel = &shared_mem[TILE_D * TILE_H * TILE_W  *Ci];


    int block_co = blockIdx.x;
    int block_d = (blockIdx.y / ((Ho + TILE_H - 1) / TILE_H)) * TILE_D;
    int block_h = ((blockIdx.y % ((Ho + TILE_H - 1) / TILE_H)) * TILE_H);
    int block_w = (blockIdx.z * TILE_W);
    int n = blockIdx.z; 


    int td = threadIdx.x;
    int th = threadIdx.y;
    int tw = threadIdx.z;

    int kernel_size = Ci * Dk * Hk * Wk;
    int kernel_base = block_co * kernel_size;
    
    for (int i = td * TILE_H * TILE_W + th * TILE_W + tw; 
         i < kernel_size; 
         i += TILE_D * TILE_H * TILE_W) {
        if (i < kernel_size) {
        }
    }
    
    __syncthreads();

    for (int d = td; d < TILE_D && block_d + d < Do; d += blockDim.x) {
        for (int h = th; h < TILE_H && block_h + h < Ho; h += blockDim.y) {
            for (int w = tw; w < TILE_W && block_w + w < Wo; w += blockDim.z) {
                float sum = 0.0f;
                
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kd = 0; kd < Dk; kd++) {
                        int id = block_d + d + kd;
                        if (id >= Di) continue;
                        
                        for (int kh = 0; kh < Hk; kh++) {
                            int ih = block_h + h + kh;
                            if (ih >= Hi) continue;
                            
                            for (int kw = 0; kw < Wk; kw++) {
                                int iw = block_w + w + kw;
                                if (iw >= Wi) continue;
                                
                                int input_idx = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;
                                
                                int kernel_idx = ci * (Dk * Hk * Wk) + kd * (Hk * Wk) + kh * Wk + kw;
                                
                                sum += input[input_idx] * shared_kernel[kernel_idx];
                            }
                        }
                    }
                }
                
                int out_d = block_d + d;
                int out_h = block_h + h;
                int out_w = block_w + w;
                
                if (out_d < Do && out_h < Ho && out_w < Wo) {
                    int output_idx = n * (Co * Do * Ho * Wo) + 
                                    block_co * (Do * Ho * Wo) + 
                                    out_d * (Ho * Wo) + 
                                    out_h * Wo + 
                                    out_w;
                    
                    output[output_idx] = sum;
                }
            }
        }
    }
}


void launch_conv3d_tiled(
    float* input , float *kernel , float *output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co, int Dk, int Hk, int Wk,
    int Do, int Ho, int Wo,
    int TILE_D , int TILE_H , int TILE_W
){
    size_t shared_mem_size = Ci * Dk * Hk * Wk * sizeof(float);

    dim3 gridDim(Ni, Co, (Do * Ho * Wo + 255) / 256);  // Grid: batch, output channels, spatial dims
    dim3 blockDim(256);  // 256 threads per block

    // Launch Kernel
    conv3d_tiled<<<gridDim, blockDim, shared_mem_size>>>(input, kernel, output, 
                                                        Ni, Ci, Di, Wi, Hi, 
                                                        Co, Dk, Hk, Wk, 
                                                        Do, Ho, Wo ,TILE_D , TILE_H , TILE_W);
    cudaDeviceSynchronize();
}