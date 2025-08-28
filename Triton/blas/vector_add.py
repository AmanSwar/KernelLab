import triton
import triton.language as tl
import torch


DEVICE = torch.device("cuda")

@triton.jit
def vector_add_kernel(
    vector_A,
    vector_B,
    vector_output,
    N,
    BLOCK_SIZE : tl.constexpr

):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    cols = block_start + tl.arange(0 , BLOCK_SIZE)

    mask = cols < N

    vector_a_elements = tl.load(vector_A + cols , mask=mask , other=None)
    vector_b_elements = tl.load(vector_B + cols , mask=mask , other=None)

    output = vector_a_elements + vector_b_elements

    tl.store(vector_output + cols , output , mask=mask)


def triton_vector_add(
        x : torch.Tensor,
        y : torch.Tensor
):
    output = torch.empty_like(x)

    N = output.numel()

    grid = lambda meta : (triton.cdiv(N , meta['BLOCK_SIZE'])) 

    vector_add_kernel[grid](
        x,
        y,
        output,
        N,
        BLOCK_SIZE=1024
    )

    return output




