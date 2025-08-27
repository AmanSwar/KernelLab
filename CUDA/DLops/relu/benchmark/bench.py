#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd


N = 16777216

# Average times in milliseconds (as reported by your benchmark)
results = {
    "Kernel": ["Naive ReLU", "Vectorized ReLU", "Optimized ReLU", "PyTorch ReLU"],
    "Avg Time (ms)": [0.0538874, 0.0592675, 0.0142752, 0.0185606],
}

# Create DataFrame
df = pd.DataFrame(results)

# Compute FLOPS for each kernel (assuming 1 FLOP per element)
# FLOPS = (N operations per iteration) / (time per iteration in seconds)
# time (s) = time (ms) / 1000, so FLOPS = N * 1000 / (time_ms)
df["FLOPS"] = df["Avg Time (ms)"].apply(lambda t: N * 1000 / t)
# Convert FLOPS to GFLOPS (billion FLOPS)
df["GFLOPS"] = df["FLOPS"] / 1e9

# Print a nicely formatted table
print("Benchmark Results:")
print(df[["Kernel", "Avg Time (ms)", "GFLOPS"]].to_string(index=False, justify="center"))

# Plotting: two bar charts side-by-side: one for time and one for GFLOPS.
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))


# Bar chart for GFLOPS
axes.bar(df["Kernel"], df["GFLOPS"], color="salmon")
axes.set_title("Performance (GFLOPS)")
axes.set_ylabel("GFLOPS")
axes.set_xticklabels(df["Kernel"], rotation=45, ha="right")

plt.tight_layout()
plt.show()
