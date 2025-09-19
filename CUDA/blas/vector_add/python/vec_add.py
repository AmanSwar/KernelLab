
import torch
import vec_add_kernels


import triton
import triton.language as tl

a = torch.rand(size=(1000000,1) , device="cuda")
b = torch.rand(size=(1000000,1) , device="cuda")

@triton.jit
def _triton_vec_add_kernel(
    vector_a,
    vector_b,
    out_vector,
    N,
    BLOCK_SIZE : tl.constexpr
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    cols_ptrs = block_start + tl.arange(0 , BLOCK_SIZE)

    rowA = tl.load(vector_a + cols_ptrs , mask=cols_ptrs < N , other=0.0)
    rowB = tl.load(vector_b + cols_ptrs , mask=cols_ptrs < N , other=0.0)

    out = rowB + rowA

    tl.store(out_vector + cols_ptrs , out, mask=cols_ptrs < N)




def launch_triton_vec_add(
    vec_a : torch.Tensor,
    vec_b : torch.Tensor,  
):
    N = vec_a.shape[1]
    out_vec = torch.zeros_like(vec_a)
    BLOCK_SIZE = 1024

    grid = lambda meta : (triton.cdiv(N , meta['BLOCK_SIZE']),)

    _triton_vec_add_kernel[grid](
        vec_a,
        vec_b,
        out_vec,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out_vec



def checker(out):

    sample_indices = [0, 1, 2, 100, 1000, 10000, 999999]
    for i in sample_indices:
        expected = a[i].item() + b[i].item()
        actual = out[i].item()
        if abs(actual - expected) > 1e-5:  # Use small epsilon for float comparison
            print(f"Error at index {i}: {actual} != {expected}")
            return False
    print("Verification passed for sampled indices")
    return True

def main():
    import time
    from torch.profiler import profile, record_function, ProfilerActivity

    _ = vec_add_kernels.multielement_vec_add(a ,b)
    _ = torch.add(a , b)
    _ = launch_triton_vec_add(a,b)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # Profile both CPU and GPU
        record_shapes=True,  # Record tensor shapes
        with_stack=True      # ESSENTIAL for flame graphs
    ) as prof:
        for i in range(10): # Run a few more iterations for a clearer profile
            with record_function("cuda kernel"): # Label your custom kernel
                c_1 = vec_add_kernels.multielement_vec_add(a ,b)

            with record_function("triton kernel"):
                c_2 = launch_triton_vec_add(a , b)

            with record_function("pytorch_add_call"): # Label the torch function
                c = torch.add(a , b)

            

    # Print the profiler results to the console
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export the trace for visualization
    prof.export_chrome_trace("trace1.json")
    print("\nProfiler trace saved to trace.json")
    # print("Upload this file to https://www.speedscope.app to view the flame graph.")

    # (Optional) Export stacks specifically for Brendan Gregg's flamegraph scripts
    # prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

    # print("torch : " , total_time_torch / 5)aawww
    # print("kernel : " , total_time_kernel / 5)
    # print(c_1 == c)
if __name__ == "__main__":
    main()

    