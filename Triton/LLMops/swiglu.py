import triton
import triton.language as tl
import torch

"""
SwiGLU -> Wx * (Swish(Vx))
Swish(y) -> y * sigmoid(y)
"""


@triton.jit
def _swigly_fwd_kernel(
    input_matrix,
    output_matrix,
    Weight1,
    Weight2,
    M , N,
    BLOCK_SIZE : tl.constexpr
):
    
    pid = tl.program_id(0) #range -> 0 * bs*M - 1

    row_in_sample = pid % M
    cols_ptr = tl.arange(0 , BLOCK_SIZE)
    mask = cols_ptr < N
    

    input_row_offset = pid * N
    input_load_ptrs = input_row_offset + cols_ptr
    #since weights are shared across bathces
    weight_row_offset = row_in_sample * N
    weight_load_ptrs = weight_row_offset + cols_ptr

    row = tl.load(input_matrix + input_load_ptrs , mask=mask , other=0.0)
    weight1_row = tl.load(Weight1 + weight_load_ptrs , mask=mask , other=0.0)
    weight2_row = tl.load(Weight2 + weight_load_ptrs , mask=mask , other=0.0)

    y = row * weight2_row

    swish_y = y * tl.sigmoid(y)

    swiglu_out = weight1_row * row * swish_y

    tl.store(output_matrix + cols_ptr , swiglu_out , mask=mask)

def launch_swigly_fwd(
    x : torch.Tensor,
    W1 : torch.Tensor,
    W2 : torch.Tensor
):
    bs , M , N = x.shape
    output_matrix = torch.zeros_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid_dim = (bs*M,)

    _swigly_fwd_kernel[grid_dim](
        input_matrix=x,
        output_matrix=output_matrix,
        Weight1=W1,
        Weight2=W2,
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        
    )

    return output_matrix



    

