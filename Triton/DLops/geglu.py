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


@triton.jit
def _geglu_bwd_kernel(
    dh_ptr,  # gradient w.r.t. output h
    e_ptr,  # first half of input (through GELU)
    g_ptr,  # second half of input (gate)
    de_ptr,  # gradient w.r.t. e (output)
    dg_ptr,  # gradient w.r.t. g (output)
    N,  # number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for GeGLU: h = g * GELU(e)

    dh/de = g * d(GELU)/de
    dh/dg = GELU(e)

    Where d(GELU)/de = 0.5 * (1 + erf(e/sqrt(2))) + 0.5 * e * (2/sqrt(pi)) * exp(-e^2/2) * (1/sqrt(2))
                     = 0.5 * (1 + erf(e/sqrt(2))) + e * exp(-e^2/2) * sqrt(1/(2*pi))
    """
    pid = tl.program_id(0)
    offset = pid * N + tl.arange(0, BLOCK_SIZE)
    mask = offset < (pid + 1) * N

    dh_row = tl.load(dh_ptr + offset, mask=mask, other=0.0)
    e_row = tl.load(e_ptr + offset, mask=mask, other=0.0)
    g_row = tl.load(g_ptr + offset, mask=mask, other=0.0)

    # Compute GELU(e)
    gelu_e = 0.5 * e_row * (1 + tl.math.erf(e_row * tl.math.rsqrt(2.0)))

    # Compute d(GELU)/de
    # GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * (2/sqrt(pi)) * exp(-x^2/2) * (1/sqrt(2))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    erf_term = tl.math.erf(e_row * tl.math.rsqrt(2.0))
    exp_term = tl.exp(-0.5 * e_row * e_row)

    dgelu_de = 0.5 * (
        1 + erf_term
    ) + 0.5 * e_row * sqrt_2_over_pi * exp_term * tl.math.rsqrt(2.0)

    # Chain rule gradients
    de_row = dh_row * g_row * dgelu_de
    dg_row = dh_row * gelu_e

    tl.store(de_ptr + offset, de_row, mask=mask)
    tl.store(dg_ptr + offset, dg_row, mask=mask)


def launch_geglu_forward(gate, up):
    batch, seq_len, hd = gate.shape
    device = gate.device
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=device)

    BLOCK_SIZE = triton.next_power_of_2(hd)
    grid = (batch * seq_len,)

    _geglu_fwd_kernel[grid](
        gate,  # e_ptr
        up,  # g_ptr
        out,  # h_ptr
        hd,  # N
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def launch_geglu_backward(dh, e, g):
    batch, seq_len, hd = e.shape
    device = e.device

    de = torch.empty_like(e)
    dg = torch.empty_like(g)

    BLOCK_SIZE = triton.next_power_of_2(hd)
    grid = (batch * seq_len,)

    _geglu_bwd_kernel[grid](
        dh,  # dh_ptr
        e,  # e_ptr
        g,  # g_ptr
        de,  # de_ptr
        dg,  # dg_ptr
        hd,  # N
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return de, dg


# Reference PyTorch implementation for comparison
def geglu_torch(gate, up):
    return up * torch.nn.functional.gelu(gate)

import math
def geglu_torch_backward(dh, gate, up):
    gelu_gate = torch.nn.functional.gelu(gate)

    # For GELU derivative, we need to compute it manually
    # GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    # GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * (2/sqrt(pi)) * exp(-x^2/2) * (1/sqrt(2))

    sqrt_2_over_pi = math.sqrt(2.0 / torch.pi)
    erf_term = torch.erf(gate / math.sqrt(2.0))
    exp_term = torch.exp(-0.5 * gate * gate)

    dgelu_dgate = (
        0.5 * (1 + erf_term) + 0.5 * gate * sqrt_2_over_pi * exp_term / math.sqrt(2.0)
    )

    dgate = dh * up * dgelu_dgate
    dup = dh * gelu_gate

    return dgate, dup


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "H"],  # Batch, Sequence, Hidden
        x_vals=[
            (1, 512, 1024),
            (2, 1024, 2048),
            (4, 2048, 4096),
        ],
        line_arg="provider",
        line_vals=["triton-fwd", "torch-fwd", "triton-bwd", "torch-bwd"],
        line_names=[
            "Triton Forward",
            "PyTorch Forward",
            "Triton Backward",
            "PyTorch Backward",
        ],
        styles=[("blue", "-"), ("red", "-"), ("blue", "--"), ("red", "--")],
        ylabel="GB/s",
        plot_name="geglu-performance",
        args={},
    )
)
def benchmark_geglu(B, S, H, provider):
    """Benchmark GeGLU forward and backward passes."""

    # Create test tensors
    gate = torch.randn(B, S, H, dtype=torch.float32, device="cuda", requires_grad=True)
    up = torch.randn(B, S, H, dtype=torch.float32, device="cuda", requires_grad=True)

    # For backward pass
    dh = torch.randn(B, S, H, dtype=torch.float32, device="cuda")

    # Calculate memory throughput
    # Forward: read 2 tensors, write 1 tensor = 3 * B * S * H * 2 bytes
    # Backward: read 3 tensors, write 2 tensors = 5 * B * S * H * 2 bytes
    numel = B * S * H
    bytes_per_element = 2  # float32

    if provider == "triton-fwd":
        fwd_bytes = 3 * numel * bytes_per_element

        def run_fn():
            return launch_geglu_forward(gate, up)

        ms = triton.testing.do_bench(run_fn)
        return fwd_bytes / ms * 1e-6  # GB/s

    elif provider == "torch-fwd":
        fwd_bytes = 3 * numel * bytes_per_element

        def run_fn():
            return geglu_torch(gate, up)

        ms = triton.testing.do_bench(run_fn)
        return fwd_bytes / ms * 1e-6  # GB/s

    elif provider == "triton-bwd":
        bwd_bytes = 5 * numel * bytes_per_element

        def run_fn():
            return launch_geglu_backward(dh, gate, up)

        ms = triton.testing.do_bench(run_fn)
        return bwd_bytes / ms * 1e-6  # GB/s

    elif provider == "torch-bwd":
        bwd_bytes = 5 * numel * bytes_per_element

        def run_fn():
            return geglu_torch_backward(dh, gate, up)

        ms = triton.testing.do_bench(run_fn)
        return bwd_bytes / ms * 1e-6  # GB/s


def test_correctness():
    """Test correctness against PyTorch implementation."""
    torch.manual_seed(42)

    # Test shapes
    test_cases = [
        (2, 128, 1024),
        (1, 512, 4096),
        (4, 256, 2048),
    ]

    for B, S, H in test_cases:
        print(f"Testing shape ({B}, {S}, {H})...")

        # Create test tensors
        gate = torch.randn(B, S, H, dtype=torch.float32, device="cuda")
        up = torch.randn(B, S, H, dtype=torch.float32, device="cuda")
        dh = torch.randn(B, S, H, dtype=torch.float32, device="cuda")

        # Forward pass
        triton_out = launch_geglu_forward(gate, up)
        torch_out = geglu_torch(gate, up)

        fwd_diff = torch.abs(triton_out - torch_out).max().item()
        print(f"  Forward max diff: {fwd_diff:.6f}")

        # Backward pass
        triton_dgate, triton_dup = launch_geglu_backward(dh, gate, up)
        torch_dgate, torch_dup = geglu_torch_backward(dh, gate, up)

        bwd_gate_diff = torch.abs(triton_dgate - torch_dgate).max().item()
        bwd_up_diff = torch.abs(triton_dup - torch_dup).max().item()

        print(f"  Backward dgate max diff: {bwd_gate_diff:.6f}")
        print(f"  Backward dup max diff: {bwd_up_diff:.6f}")

        # Check if differences are within acceptable tolerance
        tolerance = 1e-3  # float32 has limited precision
        assert fwd_diff < tolerance, f"Forward pass difference too large: {fwd_diff}"
        assert (
            bwd_gate_diff < tolerance
        ), f"Backward dgate difference too large: {bwd_gate_diff}"
        assert (
            bwd_up_diff < tolerance
        ), f"Backward dup difference too large: {bwd_up_diff}"

        print("  ✓ All tests passed!")


if __name__ == "__main__":
    print("Running correctness tests...")
    test_correctness()

    print("\nRunning benchmarks...")
    benchmark_geglu.run(save_path=".", print_data=True)
