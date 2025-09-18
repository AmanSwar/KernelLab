#include <cuda_runtime.h>

#include "naive_rms.cuh"
#include "../include/util.cuh"



int main(){
    int M = 128;
    int N = 1024;
    float eps = 1e-6f;
    test(launch_rms_bf16, "NAIVE", M, N , eps);
}
