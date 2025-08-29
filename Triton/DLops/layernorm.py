import triton
import triton.language as tl
import torch

@triton.jit
def layer_norm_fwd(
    input_matrx,
    output_matrix,
    W ,
    B,
    stride,
    M , N,
    BLOCK_SIZE : tl.constexpr
):
    
    pid = tl.program_id(axis=0)

    row_start_ptr = pid * stride + input_matrx

    col_offset = tl.arange(0 , BLOCK_SIZE)

    input_ptrs = row_start_ptr + col_offset

    mask = col_offset < N

    row = tl.load(input_ptrs , mask=mask , other=0.0)

    w = tl.load(W + col_offset , mask=mask , other=1.0)
    b = tl.load(B + col_offset , mask=mask , other=0.0)

    _row_mean = tl.sum(tl.where(mask , row , 0.0)) / N
    _numer = tl.where(mask , row-_row_mean , 0.0)

    eps = 1e-5
    _row_var = tl.sqrt((tl.sum(_numer * _numer , axis=0) / N) + eps)

    _x = _numer / _row_var

    _x = _x * w + B
    output_ptr = pid * stride + output_matrix + col_offset
    tl.store(output_ptr , _x , mask=mask)


 
def launch_ln_fwd(
    input_matrix : torch.Tensor,
    W : torch.Tensor,
    B : torch.Tensor
):
    M , N = input_matrix.shape
    output_matrix = torch.zeros_like(input_matrix)
    grid = (M,) 

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    layer_norm_fwd[grid](
        input_matrx=input_matrix,
        output_matrix=output_matrix,
        W=W,
        B=B,
        stride=N,
        M=M,
        N=N,
        BLOCK_SIZE= BLOCK_SIZE,
        num_warps=num_warps
    )

    return output_matrix



@triton.jit
def layer_norm_official_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


def launch_triton_ln_kernel_official(
        x,
        W,
        B
):
    
    y = torch.zeros_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    eps = 1e-5
    layer_norm_official_kernel[(M, )](  #
            x_arg, y, W, B, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)

    return y

