import triton
import triton.language as tl
import torch

@triton.jit
def official_softmax_kernel(
        input_matrix,
        output_matrix,
        M , N,
        input_stride,
        output_stride,
        BLOCK_SIZE : tl.constexpr,
        num_stages : tl.constexpr
):

    pid = tl.program_id(axis=0)

    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(pid , M , row_step , num_stages=num_stages):

        row_start = input_matrix + row_idx * input_stride

        cols_ptr = tl.arange(0 , BLOCK_SIZE)

        input_ptrs = row_start + cols_ptr

        mask = cols_ptr < N

        row = tl.load(input_ptrs , mask=mask , other=float('-inf'))

        row_minus_max = row - tl.max(row , axis=0)

        numer = tl.exp(row_minus_max)

        denom = tl.sum(numer , axis=0)

        softmax_output = numer / denom

        output_ptr = output_matrix + row_idx * output_stride

        tl.store(output_ptr + cols_ptr , softmax_output , mask=mask)

@triton.jit
def softmax_kernel(
    input_matrix,
    output_matrix,
    input_stride,
    output_stride,
    M , N,
    BLOCK_SIZE : tl.constexpr,
    num_stages : tl.constexpr,
):
    
    pid = tl.program_id(axis=0) #total range -> 0 to M -1

    row_start = input_matrix + pid * input_stride

    cols_ptr = tl.arange(0 , BLOCK_SIZE)
    mask = cols_ptr < N
    
    row = tl.load(row_start + cols_ptr , mask=mask , other=float('-inf'))

    _row_minus_max = row - tl.max(row , axis=0)

    numer = tl.exp(_row_minus_max)
    denom = tl.sum(numer , axis=0)

    softmax_output = numer/ denom

    output_start = output_matrix + pid * output_stride

    tl.store(output_start + cols_ptr , softmax_output , mask=mask)


DEVICE = torch.device("cuda")
device_id = DEVICE.index if DEVICE.index is not None else torch.cuda.current_device()
prop = triton.runtime.driver.active.utils.get_device_properties(device_id)


NUM_SM = prop["multiprocessor_count"]
NUM_REGS = prop["max_num_regs"]
TOTAL_SRAM_PER_SM = prop["max_shared_mem"]
WARP_SIZE = prop["warpSize"]


def launch_official_softmax_kernel(
        input_matrix : torch.Tensor
):
    
    M , N = input_matrix.shape

    BLOCK_SIZE = triton.next_power_of_2(N)

    num_warps = 4

    if BLOCK_SIZE >= 2048:
        num_warps = 8

    if BLOCK_SIZE >= 4096:
        num_warps = 16

    
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    out_matrix = torch.empty_like(input_matrix)

    grid = (min(NUM_SM * 4 , M) , )


    official_softmax_kernel[grid](
        input_matrix,
        out_matrix,
        M , N,
        input_stride=N,
        output_stride=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return out_matrix


def launch_softmax_kernel(
        input_matrix : torch.Tensor
):

    output_matrix = torch.zeros_like(input_matrix)

    M , N = input_matrix.shape

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4

    if BLOCK_SIZE >= 2048:
        num_warps = 8

    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    grid = (M,)

    softmax_kernel[grid](
        input_matrix,
        output_matrix,
        input_stride=N,
        output_stride=N,
        M=M , N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return output_matrix


if __name__ == "__main__":

    def test_softmax_kernel(
            size,
            atol=1e-3,
            rtol=1e-3,
            device=DEVICE
    ):

        torch.manual_seed(0)

        x = torch.randn(size[0] , size[1] , device=DEVICE)
        z_try = launch_softmax_kernel(x)
        z_ref = torch.softmax(x , axis=1)

        torch.testing.assert_close(z_try , z_ref , atol=atol , rtol=rtol)

        print("pass")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[128 * i for i in range(2, 20)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="softmax-performance",
            args={'M': 4096}
        )
    )
    def benchmark(M , N , provider):

        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)

        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: launch_softmax_kernel(x))
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms)

    test_softmax_kernel(size=(1832, 129))

    benchmark.run(show_plots=True)
