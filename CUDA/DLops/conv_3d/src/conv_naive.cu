#include "../include/conv_3d_kernel.h"
#include <climits>
#include <cuda_runtime.h>

// __global__ void conv3d_naive(
//     float *input,
//     float *output,
//     float *kernel,
//     int depth,
//     int height,
//     int width,
//     int kernel_depth,
//     int kernel_height,
//     int kernel_width
// ){

//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;

//     // Dimensions of output tensor
//     int out_depth = depth - kernel_depth + 1;
//     int out_height = height - kernel_height + 1;
//     int out_width = width - kernel_width + 1;

//     if (x < out_depth && y < out_height && z < out_width){
//         float sum = 0.0f;

//         for (int i = 0; i < kernel_depth; i++){
//             for (int j = 0; j < kernel_height; j++){
//                 for (int k = 0; k < kernel_width; k++){

//                     int x_idx = x + i;
//                     int y_idx = y + j;
//                     int z_idx = z + k;
                    
//                     sum += input[(x_idx * height + y_idx) * width + z_idx] * kernel[(i * kernel_height + j) * kernel_width + k];
//                 }
//             }
//         }

//         output[(x * out_height + y) * out_width + z] = sum;
//     }
// }   



__global__ void conv3d_naive(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
){
    /*
    args:
        input = input arr
        kernel = kernel
        output = output arr
        
        input params:
            Ni = Batch size  
            Ci = Channels
            Di = Depth
            Wi = Width
            Hi = Height

        kernel params:
            Dk = Depth
            Hk = Height
            Wk = Width

        output params : 
            Co = Channel
            Do = Depth
            Ho = Height
            Wo = Width
    
    */

    int n = blockIdx.x;
    int c = blockIdx.y;
    //derive d , h , w from z dimension
    // depth -> outer layer -> [[d1] [d2] ...] -> size(d1) = Ho * Wo => #di = total(di) / size(di)
    // height -> d1 = [[h1] [h2] ...] -> h1 = [[w1] [w2] .. ] 
    int d = (blockIdx.z / (Ho * Wo));
    int h = (blockDim.z % (Ho * Wo)) /  Wo;
    int w = blockIdx.z % Wo;

    // most imppppppppp -> out of bound check
    if(n >= Ni || c >= Co || d > Do || h >= Ho || w > Wo){
        return;
    }

    float sum = 0.0f;

    //iter through the kernel
    //iter through channels of input
    for(int ci = 0 ; ci < Ci ; ci++){
        //iter through depth
        for(int kd =0 ; kd < Dk ; kd++){
            //iter through height
            for(int kh = 0 ; kh < Hk ; kh++){
                //iter through width
                for(int kw = 0 ; kw < Wk ; kw++){
                    // indices in input
                    int id = d + kd;
                    int ih = h + kh;
                    int iw = w + kw;

                    
                    if(id < Di && ih < Hi && iw < Wi){

                        int input_index = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;

                        int kernel_index = c * (Ci * Dk * Hk * Wk) + ci * (Dk * Hk * Wk) + kd * (Hk * Wk) + kh * Wk + kw;


                        sum += input[input_index] * kernel[kernel_index];
                    }


                }
            }
        }
    }

    int output_index = n * (Co * Do * Ho *Wo) + c * (Do * Ho * Wo) + d * (Ho * Wo) + h * Wo + w;
    output[output_index] = sum;

}


void launch_conv3d_naive(
    float *input , float *kernel , float *output,
    int Ni , int Ci , int Di , int Wi , int Hi,
    int Co , int Dk , int Hk , int Wk,
    int Do , int Ho , int Wo
){
    dim3 gridDim(Ni , Co , (Do * Ho * Wo + 255)/256);
    dim3 blockDim(256);
    conv3d_naive<<<gridDim , blockDim>>>(input , kernel , output, Ni , Ci , Di , Wi , Hi , Co , Dk , Hk , Wk , Do , Ho , Wo);
    cudaDeviceSynchronize();


}