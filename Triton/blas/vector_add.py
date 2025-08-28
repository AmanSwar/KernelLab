import triton
import triton.language as tl
import torch


DEVICE = torch.device("cuda")

@triton.jit
def vector_add_kernel(
    vector_A,
    vector_B,
    vector_output,
    N,
    BLOCK_SIZE : tl.constexpr

):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    cols = block_start + tl.arange(0 , BLOCK_SIZE)

    mask = cols < N

    vector_a_elements = tl.load(vector_A + cols , mask=mask , other=None)
    vector_b_elements = tl.load(vector_B + cols , mask=mask , other=None)

    output = vector_a_elements + vector_b_elements

    tl.store(vector_output + cols , output , mask=mask)


def triton_vector_add(
        x : torch.Tensor,
        y : torch.Tensor
):
    output = torch.empty_like(x)

    N = output.numel()

    grid = lambda meta : (triton.cdiv(N , meta['BLOCK_SIZE']),) 

    vector_add_kernel[grid](
        x,
        y,
        output,
        N,
        BLOCK_SIZE=1024
    )

    return output


if __name__ == "__main__":

    @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['size'],
                x_vals=[2 ** i for i in range(12,28,1)],
                x_log= True,
                line_arg='provider',
                line_vals=['triton', 'torch'],
                line_names=['Triton', 'Torch'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='GB/s',
                plot_name="vec_add perf",
                args={},
            )
    )

    def benchmark(
        size,
        provider
    ):
        x = torch.rand(size , device=DEVICE , dtype=torch.float32)
        y = torch.rand(size , device=DEVICE , dtype=torch.float32)

        quantiles = [0.5 , 0.05 , 0.95]

        if provider == 'torch':
            ms , min_ms , max_ms = triton.testing.do_bench(lambda : x + y , quantiles=quantiles)

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_vector_add(x, y), quantiles=quantiles)

        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    def test_add(
            size,
            atol=1e-3,
            rtol=1e-3,
            device=DEVICE
    ):

        torch.manual_seed(696969)

        # initialize 2 tensors
        x = torch.rand(size , device=DEVICE)
        y = torch.rand(size , device=DEVICE)

        # triton output
        z_tri = triton_vector_add(x,y)
        # pytorch outpyut
        z_ref = x + y

        # verify
        torch.testing.assert_close(
            z_tri,
            z_ref,
            atol=atol,
            rtol=rtol
        )

        print("passed")

    test_add(1234556)

    benchmark.run(show_plots=True)
