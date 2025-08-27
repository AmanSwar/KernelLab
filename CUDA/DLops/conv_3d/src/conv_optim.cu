
#include <cuda_runtime.h>


template<int Dk, int Hk, int Wk, int TILE_D, int TILE_H, int TILE_W, int VEC_SIZE>
__global__ void conv3d_highly_optimized(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co, int Do, int Ho, int Wo
){
    // Instead of static arrays, allocate dynamic shared memory and partition it.
    extern __shared__ float shared_mem[];

    // First block of shared memory: kernel weights.
    float (*s_kernel)[Dk][Hk][Wk] = (float (*)[Dk][Hk][Wk]) shared_mem;
    // Second block: input tile.
    float (*s_input)[TILE_D + Dk - 1][TILE_H + Hk - 1][TILE_W + Wk - 1] = 
        (float (*)[TILE_D + Dk - 1][TILE_H + Hk - 1][TILE_W + Wk - 1])
        (shared_mem + Ci * Dk * Hk * Wk);

    // Block indices
    const int co = blockIdx.x;
    const int do_base = (blockIdx.y / ((Ho + TILE_H - 1) / TILE_H)) * TILE_D;
    const int ho_base = (blockIdx.y % ((Ho + TILE_H - 1) / TILE_H)) * TILE_H;
    const int wo_base = blockIdx.z * TILE_W;
    const int n = 0; // For simplicity, assuming batch_size = 1

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tz * blockDim.y * blockDim.x + ty * blockDim.x + tx;
    const int block_size = blockDim.x * blockDim.y * blockDim.z;

    typedef float4 float4_t;

    // --- Load kernel weights into shared memory (vectorized) ---
    #pragma unroll 4
    for (int idx = tid; idx < Ci * Dk * Hk * Wk / VEC_SIZE; idx += block_size) {
        int ci = idx / (Dk * Hk * Wk / VEC_SIZE);
        int kidx = idx % (Dk * Hk * Wk / VEC_SIZE);
        int kd = (kidx * VEC_SIZE) / (Hk * Wk);
        int khw = (kidx * VEC_SIZE) % (Hk * Wk);
        int kh = khw / Wk;
        int kw = khw % Wk;
        
        if (kw + VEC_SIZE <= Wk) {
            float4_t kernel_vec = *reinterpret_cast<const float4_t*>(&kernel[
                co * (Ci * Dk * Hk * Wk) +
                ci * (Dk * Hk * Wk) +
                kd * (Hk * Wk) +
                kh * Wk +
                kw
            ]);
            s_kernel[ci][kd][kh][kw]   = kernel_vec.x;
            s_kernel[ci][kd][kh][kw+1] = kernel_vec.y;
            s_kernel[ci][kd][kh][kw+2] = kernel_vec.z;
            s_kernel[ci][kd][kh][kw+3] = kernel_vec.w;
        } else {
            for (int i = 0; i < min(VEC_SIZE, Wk - kw); i++) {
                s_kernel[ci][kd][kh][kw+i] = kernel[
                    co * (Ci * Dk * Hk * Wk) +
                    ci * (Dk * Hk * Wk) +
                    kd * (Hk * Wk) +
                    kh * Wk +
                    kw + i
                ];
            }
        }
    }

    
    __syncthreads();
    
    // Register blocking for output values
    float r_output[TILE_D/2][TILE_H/2][TILE_W/2] = {0};
    
    // Process each input channel
    for (int ci = 0; ci < Ci; ci++) {
        // Collaborative loading of input tile
        for (int i = tid; i < (TILE_D + Dk - 1) * (TILE_H + Hk - 1) * (TILE_W + Wk - 1) / VEC_SIZE; i += block_size) {
            int d = i / ((TILE_H + Hk - 1) * (TILE_W + Wk - 1) / VEC_SIZE);
            int hw = i % ((TILE_H + Hk - 1) * (TILE_W + Wk - 1) / VEC_SIZE);
            int h = hw / ((TILE_W + Wk - 1) / VEC_SIZE);
            int w_base = (hw % ((TILE_W + Wk - 1) / VEC_SIZE)) * VEC_SIZE;
            
            int d_in = do_base + d - Dk/2;
            int h_in = ho_base + h - Hk/2;
            int w_in = wo_base + w_base - Wk/2;
            
            // Vectorized load when possible
            if (w_base + VEC_SIZE <= TILE_W + Wk - 1) {
                float4_t input_vec;
                
                if (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in >= 0 && w_in + VEC_SIZE - 1 < Wi) {
                    // Full vector is in bounds
                    input_vec = *reinterpret_cast<const float4_t*>(&input[
                        n * (Ci * Di * Hi * Wi) + 
                        ci * (Di * Hi * Wi) + 
                        d_in * (Hi * Wi) + 
                        h_in * Wi + 
                        w_in
                    ]);
                } else {
                    // Handle boundary cases
                    input_vec.x = (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in >= 0 && w_in < Wi) ? 
                        input[n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in] : 0.0f;
                    
                    input_vec.y = (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in+1 >= 0 && w_in+1 < Wi) ? 
                        input[n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in+1] : 0.0f;
                    
                    input_vec.z = (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in+2 >= 0 && w_in+2 < Wi) ? 
                        input[n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in+2] : 0.0f;
                    
                    input_vec.w = (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in+3 >= 0 && w_in+3 < Wi) ? 
                        input[n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in+3] : 0.0f;
                }
                
                // Unpack vector into shared memory
                s_input[ci][d][h][w_base] = input_vec.x;
                s_input[ci][d][h][w_base+1] = input_vec.y;
                s_input[ci][d][h][w_base+2] = input_vec.z;
                s_input[ci][d][h][w_base+3] = input_vec.w;
            } else {
                // Handle boundary cases for w
                for (int w_offset = 0; w_offset < min(VEC_SIZE, (TILE_W + Wk - 1) - w_base); w_offset++) {
                    int w = w_base + w_offset;
                    int w_in_actual = w_in + w_offset;
                    
                    s_input[ci][d][h][w] = (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in_actual >= 0 && w_in_actual < Wi) ? 
                        input[n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in_actual] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Each thread computes multiple output elements using register blocking
        #pragma unroll
        for (int do_offset = 0; do_offset < TILE_D/2; do_offset++) {
            int do_idx = do_base + tx * 2 + do_offset;
            if (do_idx >= Do) continue;
            
            #pragma unroll
            for (int ho_offset = 0; ho_offset < TILE_H/2; ho_offset++) {
                int ho_idx = ho_base + ty * 2 + ho_offset;
                if (ho_idx >= Ho) continue;
                
                #pragma unroll
                for (int wo_offset = 0; wo_offset < TILE_W/2; wo_offset++) {
                    int wo_idx = wo_base + tz * 2 + wo_offset;
                    if (wo_idx >= Wo) continue;
                    
                    float sum = 0.0f;
                    
                    // Compute the convolution for this output element with loop unrolling
                    #pragma unroll
                    for (int kd = 0; kd < Dk; kd++) {
                        #pragma unroll
                        for (int kh = 0; kh < Hk; kh++) {
                            #pragma unroll
                            for (int kw = 0; kw < Wk; kw += 4) {
                                // Compute input position in the shared memory tile
                                int d_idx = tx * 2 + do_offset + kd;
                                int h_idx = ty * 2 + ho_offset + kh;
                                int w_idx = tz * 2 + wo_offset + kw;
                                
                                // Vectorized computation when possible
                                if (kw + 4 <= Wk) {
                                    sum += s_input[ci][d_idx][h_idx][w_idx] * s_kernel[ci][kd][kh][kw];
                                    sum += s_input[ci][d_idx][h_idx][w_idx+1] * s_kernel[ci][kd][kh][kw+1];
                                    sum += s_input[ci][d_idx][h_idx][w_idx+2] * s_kernel[ci][kd][kh][kw+2];
                                    sum += s_input[ci][d_idx][h_idx][w_idx+3] * s_kernel[ci][kd][kh][kw+3];
                                } else {
                                    // Handle boundary cases
                                    for (int k = 0; k < min(4, Wk - kw); k++) {
                                        sum += s_input[ci][d_idx][h_idx][w_idx+k] * s_kernel[ci][kd][kh][kw+k];
                                    }
                                }
                            }
                        }
                    }
                    
                    r_output[do_offset][ho_offset][wo_offset] += sum;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output from registers to global memory using vectorized writes when possible
    #pragma unroll
    for (int do_offset = 0; do_offset < TILE_D/2; do_offset++) {
        int do_idx = do_base + tx * 2 + do_offset;
        if (do_idx >= Do) continue;
        
        #pragma unroll
        for (int ho_offset = 0; ho_offset < TILE_H/2; ho_offset++) {
            int ho_idx = ho_base + ty * 2 + ho_offset;
            if (ho_idx >= Ho) continue;
            
            // Try to use vectorized writes for 4 adjacent elements in the W dimension
            if (TILE_W/2 >= 4 && tz * 2 + 3 < Wo) {
                float4_t output_vec;
                output_vec.x = r_output[do_offset][ho_offset][0];
                output_vec.y = r_output[do_offset][ho_offset][1];
                output_vec.z = r_output[do_offset][ho_offset][2];
                output_vec.w = r_output[do_offset][ho_offset][3];
                
                *reinterpret_cast<float4_t*>(&output[
                    n * (Co * Do * Ho * Wo) + 
                    co * (Do * Ho * Wo) + 
                    do_idx * (Ho * Wo) + 
                    ho_idx * Wo + 
                    wo_base + tz * 2
                ]) = output_vec;
            } else {
                // Individual writes for non-vectorizable output
                #pragma unroll
                for (int wo_offset = 0; wo_offset < TILE_W/2; wo_offset++) {
                    int wo_idx = wo_base + tz * 2 + wo_offset;
                    if (wo_idx >= Wo) continue;
                    
                    output[
                        n * (Co * Do * Ho * Wo) + 
                        co * (Do * Ho * Wo) + 
                        do_idx * (Ho * Wo) + 
                        ho_idx * Wo + 
                        wo_idx
                    ] = r_output[do_offset][ho_offset][wo_offset];
                }
            }
        }
    }
}



void launch_conv3d_optim(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int Ni, int Ci, int Di, int Hi, int Wi,
    int Co , int Dk , int Hk , int Wk,
    int Do, int Ho, int Wo,
    int TILE_H , int TILE_D , int TILE_W , int VEC_SIZE, 
){
    int gridX = Co;
// - gridDim.y: tiling along Do and Ho. 
    int gridY = ((Ho + TILE_H - 1) / TILE_H) * ((Do + TILE_D - 1) / TILE_D);
    // - gridDim.z: tiling along Wo.
    int gridZ = (Wo + TILE_W - 1) / TILE_W;
    dim3 gridDim(gridX, gridY, gridZ);

    // Block dimensions:
    // Each block computes a tile; we use (TILE_D/2, TILE_H/2, TILE_W/2) threads per block.
    dim3 blockDim(TILE_D/2, TILE_H/2, TILE_W/2);

    // Compute the dynamic shared memory size needed:
    size_t shared_mem_size = Ci * (Dk * Hk * Wk 
                            + (TILE_D + Dk - 1) * (TILE_H + Hk - 1) * (TILE_W + Wk - 1))
                            * sizeof(float);

    // Launch the kernel:
    conv3d_highly_optimized<Dk, Hk, Wk, TILE_D, TILE_H, TILE_W, VEC_SIZE><<<gridDim, blockDim, shared_mem_size>>>(input, kernel, output,Ni, Ci, Di, Hi, Wi,Co, Do, Ho, Wo);
    

}