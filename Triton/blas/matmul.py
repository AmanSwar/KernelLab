import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda")


autotune_configs = [
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(configs=autotune_configs , key=["M" , "N" , "K"])
@triton.jit
def matmul_kernel(
    matrixA,
    matrixB,
    matrixC,
    M , N , K,
    stride_a_m, stride_a_k,
    stride_b_k , stride_b_n,
    stride_c_m , stride_c_n,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    GROUP_SIZE : tl.constexpr
):
    
    PID = tl.program_id(axis=0)

    #total number of PIDs required along M and N dim
    num_pid_along_m = tl.cdiv(M , BLOCK_SIZE_M) 
    num_pid_along_n = tl.cdiv(N , BLOCK_SIZE_N)

    #num of pid in grp in output matrix -> grp_size on M dim * n_dim pids
    num_pid_in_grp = GROUP_SIZE * num_pid_along_n 

    #which grip id in the output matrix
    p_group_id = PID // num_pid_in_grp

    #pid along M dim -> group_id * offset -> here offset will be grp_size
    first_pid_grp_m = p_group_id * GROUP_SIZE 

    #adjusted grp size -> incase of where first_pid_grp_m goes beyound M dim
    group_size_adj = min(num_pid_along_m - first_pid_grp_m , GROUP_SIZE)

    #pids
    PID_M = first_pid_grp_m + ((PID % num_pid_in_grp) % group_size_adj)
    PID_N = (PID % num_pid_in_grp) // group_size_adj


    #now caclulate the offset pointers 
    offsets_m = PID_M * BLOCK_SIZE_M + tl.arange(0 , BLOCK_SIZE_M)
    offsets_n = PID_N * BLOCK_SIZE_N + tl.arange(0 , BLOCK_SIZE_N)
    offsets_k = tl.arange(0 , BLOCK_SIZE_K)

    #now multiple -> so for matrix A -> [m_dim , k_dim] , matrix B -> [k_dim , n_dim]
    #we need to convert vector into matrix so add additional None dimension
    a_offsets = offsets_m[: , None] * stride_a_m + offsets_k[None , :] * stride_a_k
    b_offsets = offsets_k[: , None] * stride_b_k + offsets_n[None , :] * stride_b_n

    #acc should be fp32 for better precision
    acc = tl.zeros((BLOCK_SIZE_M , BLOCK_SIZE_N) , dtype=tl.float32)

    for k in range(0 , tl.cdiv(K , BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        current_k_offsets = k_start + offsets_k

        mask = current_k_offsets < K

        a = tl.load(matrixA + a_offsets , mask=mask[None , :] , other=0.0)
        b = tl.load(matrixB + b_offsets , mask=mask[: , None] , other=0.0)

        acc = tl.dot(a , b , acc=acc)

        a_offsets += BLOCK_SIZE_K * stride_a_k
        b_offsets += BLOCK_SIZE_K * stride_b_k


    acc = acc.to(tl.float16)

    c_offsets = stride_c_m* offsets_m[: , None] + stride_c_n * offsets_n[None , :]

    c_mask = (offsets_m[: , None] < M) & (offsets_n[None , :] < N)

    tl.store(matrixC + c_offsets , acc , mask=c_mask)


def matmul(a , b):

    a ,b = a.to(torch.float16) , b.to(torch.float16)

    (M,K) , (_ , N) = a.shape , b.shape

    c = torch.empty((M , N) , device=a.device , dtype=torch.float16)


    grid = lambda meta : (triton.cdiv(M , meta['BLOCK_SIZE_M']) * triton.cdiv(N ,meta['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        a , b ,c,
        M , N , K,
        a.stride(0), a.stride(1),
        b.stride(0) , b.stride(1),
        c.stride(0) , c.stride(1), 
    )


    return c


