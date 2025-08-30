import triton
import triton.language as tl

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt

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
    tl.store(MeanPtr + pid , row_mean ,mask=mean_var_mask)
    tl.store(RstdPtr + pid , std , mask=mean_var_mask)



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


    return output_matrix , Mean , Std
    
    


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

    tl.store(DX + pid * stride + col_ptrs , dx , mask=(pid < M))
    
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


if __name__ == "__main__":
    def check_correctness(M=4096, N=1024, eps=1e-5):
        torch.manual_seed(0)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, requires_grad=True)
        w = torch.randn((N,), device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.randn((N,), device="cuda", dtype=torch.float32, requires_grad=True)
        
        # PyTorch reference
        ln = nn.LayerNorm(N, eps=eps).cuda()
        ln.weight.data.copy_(w.detach())
        ln.bias.data.copy_(b.detach())
        y_ref = ln(x)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        dx_ref, dw_ref, db_ref = x.grad.clone(), ln.weight.grad.clone(), ln.bias.grad.clone()

        # Triton forward
        x.grad = None
        y_triton, mean, rstd = launch_ln_fwd(x, w, b)
        loss_triton = y_triton.sum()
        dy = torch.ones_like(y_triton)  # gradient of sum
        
        # Triton backward
        dx_triton, dw_triton, db_triton = launch_layernorm_bwd(dy, x, w, mean, rstd)

        # Compare
        print("Forward output close:", torch.allclose(y_triton, y_ref, atol=1e-4, rtol=1e-4))
        print("dx close:", torch.allclose(dx_triton, dx_ref, atol=1e-4, rtol=1e-4))
        print("dw close:", torch.allclose(dw_triton, dw_ref, atol=1e-4, rtol=1e-4))
        print("db close:", torch.allclose(db_triton, db_ref, atol=1e-4, rtol=1e-4))

    def benchmark(M=8192, N=1024, eps=1e-5, iters=100):
        torch.manual_seed(0)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, requires_grad=True)
        w = torch.ones((N,), device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.zeros((N,), device="cuda", dtype=torch.float32, requires_grad=True)
        
        ln = nn.LayerNorm(N, eps=eps).cuda()
        ln.weight.data.copy_(w.detach())
        ln.bias.data.copy_(b.detach())

        # Warmup
        for _ in range(10):
            y_ref = ln(x)
            y_ref.sum().backward()
            x.grad = None; ln.weight.grad = None; ln.bias.grad = None

        torch.cuda.synchronize()
        # PyTorch timing
        start = time.time()
        for _ in range(iters):
            y_ref = ln(x)
            y_ref.sum().backward()
            x.grad = None; ln.weight.grad = None; ln.bias.grad = None
        torch.cuda.synchronize()
        pyt_time = (time.time() - start) / iters

        # Triton warmup
        for _ in range(10):
            y_triton, mean, rstd = launch_ln_fwd(x, w, b)
            dx_triton, dw_triton, db_triton = launch_layernorm_bwd(torch.ones_like(y_triton), x, w, mean, rstd)
        
        torch.cuda.synchronize()
        # Triton timing
        start = time.time()
        for _ in range(iters):
            y_triton, mean, rstd = launch_ln_fwd(x, w, b)
            dx_triton, dw_triton, db_triton = launch_layernorm_bwd(torch.ones_like(y_triton), x, w, mean, rstd)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / iters
        
        time_arr = []
        time_arr.append(pyt_time*1e3)
        time_arr.append(triton_time*1e3)
        label_arr = ["Pytorch" , "Triton Kernel"]
        plt.figure(figsize=(10, 6))
        plt.bar(label_arr, time_arr, color="darkgreen")

        plt.title("Layernorm")  
        plt.xlabel("Kernels")
        plt.ylabel("Time in ms/iter (The lower the better)")
        plt.show()


        print(f"PyTorch LN: {pyt_time*1e3:.3f} ms/iter")
        print(f"Triton LN: {triton_time*1e3:.3f} ms/iter")
    

    check_correctness()
    benchmark()



    
    