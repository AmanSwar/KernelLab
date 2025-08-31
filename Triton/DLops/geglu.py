import triton
import triton.language as tl


import torch
@triton.jit
def _geglu_fwd_kernel(
    e_ptr,  # half the projection going through GELU()
    g_ptr,  # other half of the projection
    h_ptr,  # the output ptr
    N,  # number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Formula -> h = g * GELU(e)
    GELU(e) = 0.5*x * (1 + erf(x/sqrt(2)))
    """

    pid = tl.program_id(0)
    offset = pid * N + tl.arange(0, BLOCK_SIZE)
    mask = offset < (pid + 1) * N

    e_row = tl.load(e_ptr + offset, mask=mask, other=0.0)
    g_row = tl.load(g_ptr + offset, mask=mask, other=0.0)

    
    gelu_e = 0.5 * e_row * (1 + tl.math.erf(e_row * tl.math.rsqrt(2.0)))
    h_row = g_row * gelu_e

    tl.store(h_ptr + offset, h_row, mask=mask)


def launch_geglu_forward(gate, up):
    batch, seq_len, hd = gate.shape

    device = gate.device
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=device)

    BLOCK_SIZE = triton.next_power_of_2(hd)
    grid = (batch * seq_len,)  # One program per (batch, seq) position

    _geglu_fwd_kernel[grid](
        gate,  # e_ptr
        up,  # g_ptr
        out,  # h_ptr
        hd,  # N
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

