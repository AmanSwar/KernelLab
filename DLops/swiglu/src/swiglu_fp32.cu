#include <cuda_runtime.h>

/*
FORWARD PASS : 
SwiGLU(x) = (W1(x) + b1) . Swish(W2(x) + b2)
Swish = y * sigmoid(y)
sigmoid(y) = 1 / (1 + exp(-y))
y = W2x + b2

BACKWARD PASS: 
We need to return 
dL/dW1 , dL/dW2 , dL/dB1 , dl/dB2 and dL/dx

dW1 = x . dL/dv
db1 = dL/dv
dW2 = x.dL/dg
db2 = dL/dg
dL/dv = grad . Swish(g)
dL/dG = dL/dy . v . dSwish()
dSwish() = sigmoid(g) + g . d(sigmoid(g))
*/


__device__ float sigmoid(float y){

}


__global__ void swiglu_fwd_kernel(
    float* x,
    float* W1,
    float* B1,
    float* W2,
    float* B2
){

    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x

    


}


