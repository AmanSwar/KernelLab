#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cuda_runtime.h>
#include <linux/limits.h>

#define MAX_SEQ_LEN 1000

__global__
void naive_attn(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const int batch_size,
    const int seq_len,
    const int head_size,
    const float scale
){
    //batch dim
    int b = blockIdx.z;
    //query
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // head dim
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if(b >= batch_size || q_idx >= seq_len || h >= head_size){
        return;
    }


    float scores[MAX_SEQ_LEN];

    float sum_exp = 0.0f;

    for(int k_idx = 0 ; k_idx < seq_len ; k_idx++){
        float dot_prod = 0.0f;
        for(int d=0; d < head_size ; d++){
            float q_val = Q[(b * seq_len + q_idx) * head_size + d];
            float k_val = K[(b *( seq_len + k_idx)* head_size + d)];
            dot_prod += q_val * k_val;
        }

        float score = expf(dot_prod *scale);
        scores[k_idx] = score;
        sum_exp += score;
    }

    //normalize
    for(int k_idx = 0 ; k_idx < seq_len ; k_idx++){
        scores[k_idx] /= sum_exp;
    }

    float result = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float v_val = V[(b * seq_len + k_idx) * head_size + h];
        result += scores[k_idx] * v_val;
    }


    output[(b * seq_len + q_idx) * head_size + h] = result;

}


