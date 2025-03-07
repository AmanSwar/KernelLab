#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


__global__ void conv3d_shared(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
){
    extern __shared__ float shared_kernel[];

    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = (blockIdx.z / (Ho * Wo));
    int h = (blockDim.z % (Ho * Wo)) /  Wo;
    int w = blockIdx.z % Wo;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int kernel_size = Ci * Dk * Hk * Wk;
    // kernel offset
    int kernel_base = c * kernel_size;
    

    // kernel -> shared mem
    for (int i = tid; i < kernel_size; i += num_threads) {
        shared_kernel[i] = kernel[kernel_base + i];
    }
    __syncthreads();

    //boundary check
    if(n >= Ni || c > Co || d > Do || h > Ho || w > Wo){
        return;
    }

    float sum = 0.0f;


    for(int ci = 0 ; ci < Ci ; ci++){
        for(int kd = 0 ; kd < Dk ; kd++){
            for(int kh = 0 ; kh  < Hk ; kh++){
                for(int kw = 0 ; kw < Wk ; kw++){
                    int id = d + kd;
                    int ih = h + kh;
                    int iw = w + kw;

                    if(id < Di && ih < Hi && iw < Wi){

                        int input_index = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;

                        int kernel_index = c * (Ci * Dk * Hk * Wk) + ci * (Dk * Hk * Wk) + kd * (Hk * Wk) + kh * Wk + kw;

                        // pick kernel from shared mem
                        sum += input[input_index] * shared_kernel[kernel_index];
                    }
                }
            }
        }
    }

    int output_index = n * (Co * Do * Ho *Wo) + c * (Do * Ho * Wo) + d * (Ho * Wo) + h * Wo + w;
    output[output_index] = sum;

}


void launch_conv3d_shared(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
){
    size_t shared_mem_size = Ci * Dk * Hk * Wk * sizeof(float);

    dim3 gridDim(Ni, Co, (Do * Ho * Wo + 255) / 256);  // Grid: batch, output channels, spatial dims
    dim3 blockDim(256);  // 256 threads per block

    // Launch Kernel
    conv3d_shared<<<gridDim, blockDim, shared_mem_size>>>(input, kernel, output, 
                                                        Ni, Ci, Di, Wi, Hi, 
                                                        Co, Dk, Hk, Wk, 
                                                        Do, Ho, Wo);
    cudaDeviceSynchronize();


}




