import triton
import triton.language as tl

import torch

def _geglu_fwd_kernel(
    h, #the output ptr
    e, #half the projection going through GELU()
    g, #other half of the projection
    N,
    BLOCK_SIZE : tl.constexpr
):
    """
    Formula -> h = g * GELU(e)
    GELU(e) = 0.5*x * (1 + erf(x/root(2)))
    """

    pid = tl.program_id(0)
    offset = pid * N + tl.arange(0 , BLOCK_SIZE)
    mask = offset < N
    e_row = tl.load(e + offset , mask=mask , other=0.0)
    g_row = tl.load(g + offset , mask=mask , other=0.0)
    

    f_row = 0.5 * e_row + (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1)
    h_row = f_row * g_row

    tl.store(h + offset , h_row , mask=mask)


