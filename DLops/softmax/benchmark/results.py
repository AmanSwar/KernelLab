# Kernel names and their average execution times in milliseconds.
kernel_names = [
    "Naive Softmax",
    "Shared Memory Softmax",
    "Warp Optimized Softmax",
    "Block Optimized Softmax",
    "SOTA Optimized Softmax",
    "Torch Softmax"
]

# Average execution times in ms.
times_ms = [5.82393, 0.996116, 1.83028, 1.78251, 1.49085, 0.841318]

# Problem dimensions.
N = 4096
D = 4096

# Total FLOPs assuming 5 operations per element.
total_flops = 5 * N * D  # 5 * 4096 * 4096

# Compute GFLOPs for each kernel.
# GFLOPs = total_flops / (time_in_seconds * 1e9)
# Since time_in_seconds = time_ms / 1000, we have:
# GFLOPs = total_flops / (time_ms * 1e6)
gflops = [total_flops / (t * 1e6) for t in times_ms]

# Print header.
header = f"{'Kernel':<30}{'Time (ms)':>15}{'GFLOPs':>15}"
separator = "-" * len(header)
print(header)
print(separator)

# Print each row.
for name, t, g in zip(kernel_names, times_ms, gflops):
    print(f"{name:<30}{t:>15.5f}{g:>15.2f}")
