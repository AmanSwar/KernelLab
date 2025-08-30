import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_fwd_kernel(
    input_matrix,
    output_matrix,
    WeightPtr,
    BiasPtr,
    MeanPtr,
    RstdPtr,
    stride,
    M , N,
    eps,
    BLOCK_SIZE : tl.constexpr
):
    
    pid = tl.program_id(axis=0)

    row_start = pid * stride + input_matrix
    cols_ptrs = tl.arange(0 , BLOCK_SIZE)

    input_ptrs = row_start + cols_ptrs
    mask = cols_ptrs < N
    
    #load everything
    row = tl.load(input_ptrs ,mask=mask , other=0.0)
    weight = tl.load(WeightPtr + cols_ptrs , mask=mask , other=0.0)
    bias = tl.load(BiasPtr + cols_ptrs , mask=mask , other=0.0)

    #row mean should be stored
    row_mean = (tl.sum(tl.where(mask , row , 0.0))) / N

    _numer = tl.where(mask , row - row_mean , 0.0)

    std = 1 / (tl.sqrt((tl.sum(_numer * _numer , axis=0) / N) + eps))

    _out = _numer * std

    xhat = weight * _out +  bias

    output_ptrs = (output_matrix + pid * stride) + cols_ptrs
    tl.store(output_ptrs , xhat , mask=mask)
    mean_var_mask = pid < M
    tl.store(MeanPtr + pid , row_mean ,mask=mask)
    tl.store(RstdPtr + pid , std , mask=mask)



def launch_ln_fwd(
    input_matrix : torch.Tensor,
    W : torch.Tensor,
    B : torch.Tensor
):
    M , N = input_matrix.shape
    
    output_matrix = torch.zeros_like(input_matrix)
    Mean = torch.empty((M,) , dtype=torch.float32 , device=input_matrix.device)
    Std = torch.empty((M,) , dtype=torch.float32 , device=input_matrix.device)
    
    grid = (M,) 

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    layernorm_fwd_kernel[grid](
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        WeightPtr=W,
        BiasPtr=B,
        MeanPtr=Mean,
        RstdPtr=Std,
        stride=N,
        M=M,
        N=N,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4
    )


    return output_matrix
    
    


@triton.jit
def layernorm_bwd_kernel(
    DX, #dx 
    DY, #incomming gradients
    DW, #weight grads pointers
    DB, #bias grads pointers
    input_matrix_ptr, #input matrix ptr
    MeanPtr,
    RstdPtr,
    WeightPtr,
    stride,
    M , N,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride + input_matrix_ptr
    col_ptrs = tl.arange(0 , BLOCK_SIZE)
    
    mask_row = col_ptrs < N
    
    #load the particular row
    row = tl.load(row_start + col_ptrs , mask_row , other=0.0)
    
    #load the corresponding mean and std values
    mean = tl.load(MeanPtr + pid , mask=(pid < M) , other=0.0)
    rstd = tl.load(RstdPtr + pid , mask=(pid < M) , other=0.0)

    #load the icomming gradients
    dy = tl.load(DY + pid * stride + col_ptrs, mask=mask_row , other=0.0)
    
    #load the weight value
    weight = tl.load(WeightPtr + col_ptrs , mask=mask_row , other=1.0)
    
    xhat = (tl.where(mask_row , row - mean , 0.0)) * rstd


    dweight = tl.sum(dy * xhat , axis=0)
    dbias = tl.sum(dy , axis=0)

    dx_hat = dy * weight

    #calculation of dx -> 
    # rstd * (dx_hat - mean(dx_hat)) - x_hat * mean((dx_hat * x_hat))

    dx_hat_mean = tl.sum(dx_hat , axis=0) / N
    mean_dxhat_xhat = tl.sum(dx_hat * xhat , axis=0) / N

    dx = rstd * (dx_hat - dx_hat_mean - xhat * mean_dxhat_xhat)

    tl.store(DX + pid * stride + cols , dx , mask=(pid < M))
    
    tl.atomic_add(DW+col_ptrs , dy*xhat , mask=mask_row)
    tl.atomic_add(DB + col_ptrs , dy , mask=mask_row)


def launch_layernorm_bwd(
    dy : torch.Tensor,
    x : torch.Tensor,
    w : torch.Tensor,
    mean : torch.Tensor,
    rstd : torch.Tensor
):
    M , N = x.shape
    
    dx = torch.empty_like(x)
    dweight = torch.empty_like(w)
    dbias = torch.zeros_like(w)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    num_warps = min(max(BLOCK_SIZE // 256 , 1) , 8)
    layernorm_bwd_kernel[grid](
        DX=dx,
        DY=dy,
        DW=dweight,
        DB=dbias,
        input_matrix_ptr=x,
        MeanPtr=mean,
        RstdPtr=rstd,
        WeightPtr=w,
        stride=N,
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4
    )

    return dx , dweight , dbias



    





    
    