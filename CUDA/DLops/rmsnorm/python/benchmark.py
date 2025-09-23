import os
import subprocess
import time
import math
import statistics

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rmsnorm_kernel_vec import rmsnorm_kernel_vec 
from rmsnorm_kernel_naive import rmsnorm_kernel_naive as rmsnorm_kernel


TEST_SHAPES = [
    (32, 1, 128),
    (32, 4, 128),
    (32, 16, 256),
    (16, 64, 512),
    (8, 128, 1024),
    (4, 256, 2048),
    (2, 512, 4096),
]

WARMUP = 10
REPEATS = 50


device = torch.device("cuda")
def rmsnorm_reference(input_bf16, weight_bf16, eps=1e-6):

    bs , seq_len , embed_dim = input_bf16.shape
    M = bs * seq_len

    rms = torch.nn.RMSNorm(embed_dim , device=device)
    # return torch.nn.functional.rms_norm(input_bf16 , normalized_shape=(embed_dim,) , weight=weight_bf16 , eps=eps)
    return rms(input_bf16)


def compute_flops(M, N):
    # total flops = M*N*(4 + 3/N) = 4*M*N + 3*M
    return 4.0 * M * N + 3.0 * M

def time_fn(fn, *args, warmup=WARMUP, repeats=REPEATS):
    # fn should be a callable that runs the op once; ensure it synchronizes CUDA internally.
    # Do warmup
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times), statistics.mean(times), min(times)

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = False

    results = []  # list of dicts

    for (bs, seq_len, embed_dim) in TEST_SHAPES:
        print(f"\nBenchmarking shape bs={bs}, seq_len={seq_len}, embed_dim={embed_dim}")
        M = bs * seq_len
        N = embed_dim

        try:
            x = torch.randn(bs, seq_len, embed_dim, dtype=torch.bfloat16, device=device)
            w = torch.randn(embed_dim, dtype=torch.bfloat16, device=device)
            use_bf16 = True
        except Exception:
            print("bfloat16 allocation failed; falling back to float32 for reference only.")
            x = torch.randn(bs, seq_len, embed_dim, dtype=torch.float32, device=device)
            w = torch.randn(embed_dim, dtype=torch.float32, device=device)
            use_bf16 = False

        def ref_call(inp, wt, eps=1e-6):
            out = rmsnorm_reference(inp.to(torch.bfloat16), wt.to(torch.bfloat16), eps)
            torch.cuda.synchronize()
            return out

        def fused_call(inp, wt, eps=1e-6):
            out = rmsnorm_kernel_vec(inp, wt, float(eps))
            torch.cuda.synchronize()
            return out

        def fused_call_2(inp, wt, eps=1e-6):
            out = rmsnorm_kernel(inp, wt, float(eps))
            torch.cuda.synchronize()
            return out

        if use_bf16:
            out_ref = rmsnorm_reference(x, w)
            try:
                out_fused = fused_call(x, w)
                out_fused2 = fused_call_2(x , w)
            except Exception as e:
                print("Fused kernel call failed on this input:", e)
                out_fused = None

            if out_fused is not None:
                diff = out_ref.to(torch.float32) - out_fused.to(torch.float32)
                max_abs_err = diff.abs().max().item()
                mean_abs_err = diff.abs().mean().item()
                print(f"Correctness Vectorized: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}")
            else:
                print("Skipping correctness compare because fused kernel call failed.")

            if out_fused is not None:
                diff = out_ref.to(torch.float32) - out_fused2.to(torch.float32)
                max_abs_err = diff.abs().max().item()
                mean_abs_err = diff.abs().mean().item()
                print(
                    f"Correctness naive: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
                )
            else:
                print("Skipping correctness compare because fused kernel call failed.")
        else:
            print("Skipped fused kernel correctness check (no bfloat16 support); reference still runs.")

        # Benchmark reference (we'll time the PyTorch float32 reference)
        print("Timing reference (PyTorch float32) ...")
        if use_bf16:
            median_ref, mean_ref, min_ref = time_fn(lambda: ref_call(x, w))
        else:
            x_f = x.to(torch.float32)
            w_f = w.to(torch.float32)
            def ref_call_f32():
                mean_sq = x_f.pow(2).mean(dim=-1, keepdim=True)
                rms = torch.sqrt(mean_sq + 1e-6)
                out = x_f / rms * w_f.view(*([1] * (x_f.dim() - 1)), -1)
                torch.cuda.synchronize()
                return out
            median_ref, mean_ref, min_ref = time_fn(lambda: rmsnorm_reference(x ,w))

        fused_available = True
        try:
            _ = fused_call(x, w)
            median_fused, mean_fused, min_fused = time_fn(lambda: fused_call(x, w))
        except Exception as e:
            print("Fused kernel unavailable or failed to run:", e)
            fused_available = False
            median_fused = mean_fused = min_fused = float('nan')

        try:
            _ = fused_call(x, w)
            median_fused_2, mean_fused_2, min_fused_2 = time_fn(lambda: fused_call_2(x, w))
        except Exception as e:
            print("Fused kernel unavailable or failed to run:", e)
            fused_available = False
            median_fused = mean_fused = min_fused = float("nan")

        # Compute GFLOPS for each
        flops = compute_flops(M, N)
        gflops_ref = (flops / median_ref) / 1e9
        gflops_fused = (flops / median_fused) / 1e9 if fused_available else float('nan')
        gflops_fused_2 = (flops / median_fused_2) / 1e9 if fused_available else float('nan')

        print(f"Shape M={M}, N={N}, flops per call ~ {flops:.2f}")
        print(f"Reference median time: {median_ref*1000:.3f} ms  -> {gflops_ref:.3f} GFLOPS")
        if fused_available:
            print(f"vectorized median time:     {median_fused*1000:.3f} ms  -> {gflops_fused:.3f} GFLOPS")
            print(f"naive median time:     {median_fused_2*1000:.3f} ms  -> {gflops_fused:.3f} GFLOPS")
        else:
            print("Fused kernel not available for this config.")

        results.append({
            "bs": bs, "seq_len": seq_len, "embed_dim": embed_dim,
            "M": M, "N": N,
            "flops": flops,
            "ref_time_s": median_ref, "ref_gflops": gflops_ref,
            "fused_time_s": median_fused, "fused_gflops": gflops_fused,
            "fused_available": fused_available,
        })

    plt.figure(figsize=(8,5))
    embed_dims = sorted(list({r["N"] for r in results}))
    seqs = sorted(list({r["seq_len"] for r in results}))
    markers = ['o','s','^','x','d','*','+']
    for i, seq in enumerate(seqs):
        x_vals = []
        y_ref = []
        y_fused = []
        for r in sorted([rr for rr in results if rr["seq_len"]==seq], key=lambda z:z["N"]):
            x_vals.append(r["N"])
            y_ref.append(r["ref_gflops"])
            y_fused.append(r["fused_gflops"] if r["fused_available"] else np.nan)
        plt.plot(x_vals, y_ref, marker=markers[i%len(markers)], linestyle='-', label=f'ref seq={seq}')
        plt.plot(x_vals, y_fused, marker=markers[(i+1)%len(markers)], linestyle='--', label=f'fused seq={seq}')

    plt.xlabel("embed_dim (N)")
    plt.ylabel("GFLOPS")
    plt.title("RMSNorm GFLOPS: reference (PyTorch) vs fused kernel")
    plt.grid(True)
    plt.legend()
    out_png = "rmsnorm_bench.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")

    # print a summary table
    print("\nSummary:")
    print("{:>8} {:>8} {:>8} {:>12} {:>12} {:>12}".format("bs","seq","N","ref_time_ms","ref_GFLOPS","fused_GFLOPS"))
    for r in results:
        print("{:8d} {:8d} {:8d} {:12.3f} {:12.3f} {:12.3f}".format(
            r["bs"], r["seq_len"], r["N"],
            r["ref_time_s"]*1000, r["ref_gflops"], r["fused_gflops"] if r["fused_available"] else float('nan')
        ))

if __name__ == "__main__":
    main()
