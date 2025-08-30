import triton
import triton.language as tl
import torch
import time

ROPE_GROUP_SIZE: int = 4


@triton.jit
def _rope_embed(
    Q,
    q_stride: tl.constexpr,
    cos,
    cos_stride: tl.constexpr,
    sin,
    sin_stride: tl.constexpr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ROPE_GROUP_SIZE: int = 4
    row_start = tl.program_id(0)
    group_head_position = tl.program_id(axis=1)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin_vals = tl.load(
        sin + (row_start % seqlen) * sin_stride + col_offsets, mask=mask, other=0.0
    )
    cos_vals = tl.load(
        cos + (row_start % seqlen) * cos_stride + col_offsets, mask=mask, other=0.0
    )

    if BACKWARD_PASS:
        sin_vals = -sin_vals

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_start * q_stride + k * head_dim + col_offsets
        offs_q2 = row_start * q_stride + k * head_dim + col_offsets + half_head_dim

        Q1 = tl.load(Q + offs_q1, mask=mask, other=0.0).to(cos_vals.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0.0).to(cos_vals.dtype)

        new_Q1 = Q1 * cos_vals - Q2 * sin_vals
        new_Q2 = Q2 * cos_vals + Q1 * sin_vals

        tl.store(Q + offs_q1, new_Q1, mask=mask)
        tl.store(Q + offs_q2, new_Q2, mask=mask)


_rope_embedding = triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: bool(args["BACKWARD_PASS"]),
    }
)(_rope_embed)


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape

        Q = Q.contiguous().view(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = Q.shape

        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
        num_warps = 4  

        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        _rope_embedding[(n_rows, n_groups)](
            Q,
            Q.stride(0),
            cos,
            cos.stride(0),
            sin,
            sin.stride(0),
            seq_len,
            head_dim,
            n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        ctx.seq_len = seq_len
        ctx.head_dim = head_dim
        ctx.n_heads = n_heads

        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.contiguous().view(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = dY.shape

        _rope_embedding[(n_rows, ctx.n_groups)](
            dY,
            dY.stride(0),
            ctx.cos,
            ctx.cos.stride(0),
            ctx.sin,
            ctx.sin.stride(0),
            ctx.seq_len,
            ctx.head_dim,
            ctx.n_heads,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None  


def torch_rope_embedding(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference PyTorch implementation of RoPE embedding"""
    batch, seq_len, n_heads, head_dim = Q.shape
    half_dim = head_dim // 2

    # Split into first and second half
    Q1 = Q[..., :half_dim]  # Shape: [batch, seq_len, n_heads, half_dim]
    Q2 = Q[..., half_dim:]  # Shape: [batch, seq_len, n_heads, half_dim]

    # Expand cos and sin to match Q dimensions
    cos_expanded = (
        cos[:seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
    )  # [1, seq_len, 1, half_dim]
    sin_expanded = (
        sin[:seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
    )  # [1, seq_len, 1, half_dim]

    # Apply rotation
    Q_rotated = torch.cat(
        [Q1 * cos_expanded - Q2 * sin_expanded, Q2 * cos_expanded + Q1 * sin_expanded],
        dim=-1,
    )

    return Q_rotated


def create_rope_tables(
    seq_len: int, head_dim: int, theta: float = 10000.0, device: str = "cuda"
):
    """Create RoPE cos/sin tables"""
    half_dim = head_dim // 2
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    )

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs_grid = torch.outer(t, freqs)

    cos_table = torch.cos(freqs_grid)
    sin_table = torch.sin(freqs_grid)

    return cos_table, sin_table


def test_correctness():
    """Test correctness of Triton implementation against PyTorch reference"""
    print("Testing correctness...")

    # Test parameters
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    device = "cuda"

    # Create test data
    Q = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    cos_table, sin_table = create_rope_tables(seq_len, head_dim, device=device)

    # PyTorch reference
    Q_ref = Q.clone().detach().requires_grad_(True)
    output_ref = torch_rope_embedding(Q_ref, cos_table, sin_table)

    # Triton implementation
    Q_triton = Q.clone().detach().requires_grad_(True)
    output_triton = Fast_RoPE_Embedding.apply(Q_triton, cos_table, sin_table)

    # Check forward pass
    forward_error = torch.max(torch.abs(output_ref - output_triton))
    print(f"Forward pass max error: {forward_error:.6f}")

    # Check backward pass
    grad_output = torch.randn_like(output_ref)

    # Reference backward
    output_ref.backward(grad_output, retain_graph=True)
    grad_ref = Q_ref.grad.clone()

    # Triton backward
    Q_triton.grad = None
    output_triton.backward(grad_output)
    grad_triton = Q_triton.grad

    backward_error = torch.max(torch.abs(grad_ref - grad_triton))
    print(f"Backward pass max error: {backward_error:.6f}")

    # Tolerance check
    forward_ok = forward_error < 1e-3
    backward_ok = backward_error < 1e-3

    print(f"Forward pass: {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print(f"Backward pass: {'✓ PASS' if backward_ok else '✗ FAIL'}")

    return forward_ok and backward_ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],  # Argument names to use as x-axis
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],  # Different sequence lengths
        line_arg="provider",  # Argument name whose value corresponds to a different line
        line_vals=["torch", "triton"],  # Label for each line
        line_names=["PyTorch", "Triton"],  # Human-readable names for each line
        styles=[("blue", "-"), ("red", "-")],  # Line styles
        ylabel="Execution Time (ms)",  # Y-axis label
        plot_name="rope-embedding-performance",  # Name of the plot
        args={"batch_size": 4, "n_heads": 8, "head_dim": 64},  # Fixed arguments
    )
)
def benchmark_rope_forward(
    seq_len, batch_size, n_heads, head_dim, provider, device="cuda"
):
    """Benchmark forward pass performance"""
    # Create test data
    Q = torch.randn(
        batch_size, seq_len, n_heads, head_dim, device=device, dtype=torch.float16
    )
    cos_table, sin_table = create_rope_tables(seq_len, head_dim, device=device)

    # Warmup
    for _ in range(10):
        if provider == "torch":
            _ = torch_rope_embedding(Q, cos_table, sin_table)
        elif provider == "triton":
            _ = Fast_RoPE_Embedding.apply(Q, cos_table, sin_table)

    # Timing function
    def time_fn():
        torch.cuda.synchronize()
        start = time.time()

        if provider == "torch":
            output = torch_rope_embedding(Q, cos_table, sin_table)
        elif provider == "triton":
            output = Fast_RoPE_Embedding.apply(Q, cos_table, sin_table)

        torch.cuda.synchronize()
        end = time.time()
        return (end - start) * 1000  # Convert to milliseconds

    # Multiple runs for accurate timing
    times = [time_fn() for _ in range(100)]
    return sum(times) / len(times)

if __name__ == "__main__":
    test_correctness()
    benchmark_rope_forward.run(show_plots=True)
