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

@triton.jit
def _geglu_bwd_kernel(DW, e, g, N, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * N + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    dw_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row

    h_row = f_row * g_row
    df_row = dw_row * f_row

    dg_row = dw_row * g_row

    _SQRT_CONST = 0.3989422804014327
    df_de = f_partial_row + _SQRT_CONST * e_row * tl.exp(-0.5 * e_row * e_row)

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(dw_row.dtype)

    tl.store(DW + offsets, h_row, mask=mask)  # h  = f * g
    tl.store(e + offsets, df_row, mask=mask)  # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)


def launch_geglu_backward(DW, e , g):
    batch_seq_len, hd = e.shape
    N = e.numel()

    BLOCK_SIZE = triton.next_power_of_2(hd)
    grid = (batch_seq_len,) 

    _geglu_bwd_kernel[grid](
        DW,  # e_ptr
        e,  # g_ptr
        g,  # h_ptr
        N,  # N
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return DW, e, g


def geglu_torch(gate, up):
    return up * torch.nn.functional.gelu(gate)


def geglu_torch_backward(dh, gate, up):
    gelu_gate = torch.nn.functional.gelu(gate)


    sqrt_2_over_pi = (2.0 / torch.pi).sqrt()
    erf_term = torch.erf(gate / (2.0).sqrt())
    exp_term = torch.exp(-0.5 * gate * gate)

    dgelu_dgate = (
        0.5 * (1 + erf_term) + 0.5 * gate * sqrt_2_over_pi * exp_term / (2.0).sqrt()
    )

    dgate = dh * up * dgelu_dgate
    dup = dh * gelu_gate

    return dgate, dup
