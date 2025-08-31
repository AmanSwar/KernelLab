import triton
import triton.language as tl
import triton.testing as tt
import torch
import time

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

@triton.jit
def _swiglu_bwd_kernel(
    input_matrix,            
    doutput_matrix,        
    Weight1,
    Weight2,
    Dx,
    Dw1,         
    Dw2,         
    bs,
    M,               
    N,               
    BLOCK_SIZE: tl.constexpr
):

    pid = tl.program_id(0) 
    row_in_sample = pid % M
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    base = pid * N
    ptrs = base + offsets

    x_row = tl.load(input_matrix + ptrs, mask=mask, other=0.0)
    dout_row = tl.load(doutput_matrix + ptrs, mask=mask, other=0.0)

    w_row_offset = row_in_sample * N
    w_ptrs = w_row_offset + offsets
    w1_row = tl.load(Weight1 + w_ptrs, mask=mask, other=0.0)
    w2_row = tl.load(Weight2 + w_ptrs, mask=mask, other=0.0)

    y = w2_row * x_row
    sig_out = tl.sigmoid(y)
    swish_y = y  * sig_out

    dswish_dy = sig_out + y * sig_out * (1 - sig_out)

    dx_row = dout_row * w1_row * (swish_y + x_row * dswish_dy * w2_row)
    dW1_contrib = dout_row * x_row * swish_y
    dW2_contrib = dout_row * w1_row * (x_row * x_row) * dswish_dy

    tl.store(Dx + ptrs, dx_row, mask=mask)
    tl.atomic_add(Dw1 + w_ptrs, dW1_contrib, mask=mask)
    tl.atomic_add(Dw2 + w_ptrs, dW2_contrib, mask=mask)


def launch_swiglu_bwd(
    x: torch.Tensor, 
    dout: torch.Tensor, 
    W1: torch.Tensor, 
    W2: torch.Tensor
):
    bs, M, N = x.shape

    dx = torch.empty_like(x)
    dW1 = torch.zeros_like(W1)  # shape (M,N)
    dW2 = torch.zeros_like(W2)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (bs * M,)

    _swiglu_bwd_kernel[grid](
        x, dout, W1, W2, dx, dW1, dW2, bs, M, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return dx, dW1, dW2


def swiglu_fwd_torch(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor):
    # x: (bs, M, N); W1,W2: (M,N)
    y = x * W2.unsqueeze(0)  # broadcast to (bs,M,N)
    swish_y = y * torch.sigmoid(y)
    out = W1.unsqueeze(0) * x * swish_y
    return out, y, swish_y


def swiglu_bwd_torch(
    dout: torch.Tensor,
    x: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    y=None,
    swish_y=None,
):
    if y is None:
        y = x * W2.unsqueeze(0)
    if swish_y is None:
        swish_y = y * torch.sigmoid(y)
    sig = torch.sigmoid(y)
    dswish_dy = sig + y * sig * (1 - sig)
    dW1 = (dout * x * swish_y).sum(dim=0)  # (M,N)
    dW2 = (dout * W1.unsqueeze(0) * (x * x) * dswish_dy).sum(dim=0)
    dx = dout * W1.unsqueeze(0) * (swish_y + x * dswish_dy * W2.unsqueeze(0))
    return dx, dW1, dW2


def sync():
    torch.cuda.synchronize()


def time_fn(fn, niter=50, warmup=10):
   
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(niter):
        fn()
    sync()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / niter
    return avg_ms


def check_correctness(bs=4, M=8, N=32, dtype=torch.float32, atol=1e-5):
    device = torch.device("cuda")
    x = torch.randn(bs, M, N, device=device, dtype=dtype)
    W1 = torch.randn(M, N, device=device, dtype=dtype)
    W2 = torch.randn(M, N, device=device, dtype=dtype)

    # forward
    t_out, y, swish_y = swiglu_fwd_torch(x, W1, W2)
    triton_out = launch_swigly_fwd(
        x.contiguous(), W1.contiguous(), W2.contiguous()
    )
    if not torch.allclose(t_out, triton_out, atol=atol, rtol=1e-4):
        print("Forward mismatch!")
        print("max abs diff:", (t_out - triton_out).abs().max().item())
        return False
    # backward
    dout = torch.randn_like(t_out)
    dx_ref, dW1_ref, dW2_ref = swiglu_bwd_torch(dout, x, W1, W2, y=y, swish_y=swish_y)
    dx_tr, dW1_tr, dW2_tr = launch_swiglu_bwd(
        x.contiguous(), dout.contiguous(), W1.contiguous(), W2.contiguous()
    )
    ok_dx = torch.allclose(dx_ref, dx_tr, atol=atol, rtol=1e-4)
    ok_w1 = torch.allclose(dW1_ref, dW1_tr, atol=atol, rtol=1e-4)
    ok_w2 = torch.allclose(dW2_ref, dW2_tr, atol=atol, rtol=1e-4)
    if not (ok_dx and ok_w1 and ok_w2):
        print("Backward mismatch!")
        print("dx max diff:", (dx_ref - dx_tr).abs().max().item())
        print("dW1 max diff:", (dW1_ref - dW1_tr).abs().max().item())
        print("dW2 max diff:", (dW2_ref - dW2_tr).abs().max().item())
        return False
    return True



@tt.perf_report(
    tt.Benchmark(
        x_names=["bs", "M", "N"],
        x_vals=[(32, 64, 512), (32, 128, 1024), (64, 128, 1024)],
        line_arg="op",
        line_vals=["triton", "torch"],
        line_names=["triton", "torch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="swiglu_fwd_benchmark",
        args={}
    )
)
def perf_forward(bs, M, N, op):
    device = torch.device("cuda")
    dtype = torch.float32
    x = torch.randn(bs, M, N, device=device, dtype=dtype).contiguous()
    W1 = torch.randn(M, N, device=device, dtype=dtype).contiguous()
    W2 = torch.randn(M, N, device=device, dtype=dtype).contiguous()

    if op == "triton":
        fn = lambda: launch_swigly_fwd(x, W1, W2)
    else:
        fn = lambda: swiglu_fwd_torch(x, W1, W2)[0]

  
    def run_once():
        out = fn()
    
        sync()
        return out

    avg_ms = time_fn(run_once, niter=40, warmup=10)
    return avg_ms


@tt.perf_report(
    tt.Benchmark(
        x_names=["bs", "M", "N"],
        x_vals=[(32, 64, 512), (32, 128, 1024), (64, 128, 1024)],
        line_arg="op",
        line_vals=["triton", "torch"],
        line_names=["triton", "torch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="swiglu_bwd_benchmark",
        args={}
    )
)
def perf_backward(bs, M, N, op):
    device = torch.device("cuda")
    dtype = torch.float32
    x = torch.randn(bs, M, N, device=device, dtype=dtype).contiguous()
    W1 = torch.randn(M, N, device=device, dtype=dtype).contiguous()
    W2 = torch.randn(M, N, device=device, dtype=dtype).contiguous()
    dout = torch.randn(bs, M, N, device=device, dtype=dtype).contiguous()

    if op == "triton":
        fn = lambda: launch_swiglu_bwd(x, dout, W1, W2)
    else:
        fn = lambda: swiglu_bwd_torch(dout, x, W1, W2)[0:3]  # returns dx,dW1,dW2

    def run_once():
        out = fn()
        sync()
        return out

    avg_ms = time_fn(run_once, niter=40, warmup=10)
    return avg_ms


if __name__ == "__main__":
    print("Checking correctness (small)...")
    ok = check_correctness(bs=4, M=8, N=64)
    print("Correctness:", ok)
    print(
        "\nRunning forward perf report (this will open a plot if you're in an env that supports it)..."
    )
    perf_forward.run(show_plots=True)  
    print("\nRunning backward perf report...")
    perf_backward.run(show_plots=True)
