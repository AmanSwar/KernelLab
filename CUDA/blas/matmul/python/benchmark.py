import torch
import matmul_kernel
import time
import matplotlib.pyplot as plt
import numpy as np

# Fixed size of 512
SIZE = 512
FLOP = 2 * SIZE * SIZE * SIZE

def benchmark(num_runs=5):
    torch_times = []
    custom_times = []
    
    for i in range(num_runs):
        # Initialize matrices
        mata = torch.rand(size=(SIZE, SIZE), device="cuda")
        matb = torch.rand(size=(SIZE, SIZE), device="cuda")
        
        # Warmup
        _ = torch.matmul(mata, matb)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch matmul
        torch.cuda.synchronize()
        st_2 = time.monotonic()
        matd = torch.matmul(mata, matb)
        torch.cuda.synchronize()
        et_2 = time.monotonic() - st_2
        
        # Benchmark custom kernel
        try:
            torch.cuda.synchronize()
            st_1 = time.monotonic()
            matc = matmul_kernel.matmul_kernel(mata, matb)
            torch.cuda.synchronize()
            et_1 = time.monotonic() - st_1
            
            # Calculate TFLOPS
            torch_tflops = (FLOP/et_2)*1e-12
            custom_tflops = (FLOP/et_1)*1e-12
            
            # Print results for this run
            print(f"Run {i+1}:")
            print(f"PyTorch matmul: {et_2:.6f}s, {torch_tflops:.2f} TFLOPS")
            print(f"Custom kernel: {et_1:.6f}s, {custom_tflops:.2f} TFLOPS")
            print(f"Speedup: {et_2/et_1:.2f}x\n")
            
            # Store times
            torch_times.append(et_2)
            custom_times.append(et_1)
            
        except Exception as e:
            print(f"Run {i+1}:")
            print(f"PyTorch matmul: {et_2:.6f}s")
            print(f"Custom kernel: ERROR - {str(e)}\n")
    
    return torch_times, custom_times

def create_visualization(torch_times, custom_times):
    if not torch_times or not custom_times:
        print("No successful benchmark runs to visualize")
        return
    
    # Calculate average times and TFLOPS
    avg_torch_time = np.mean(torch_times)
    avg_custom_time = np.mean(custom_times)
    avg_torch_tflops = (FLOP/avg_torch_time)*1e-12
    avg_custom_tflops = (FLOP/avg_custom_time)*1e-12
    speedup = avg_torch_time / avg_custom_time
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar colors
    colors = ['#3498db', '#e74c3c']
    
    # Performance comparison
    labels = ['PyTorch', 'Custom']
    times = [avg_torch_time, avg_custom_time]
    tflops = [avg_torch_tflops, avg_custom_tflops]
    
    # Time comparison
    ax1.bar(labels, times, color=colors)
    ax1.set_title(f'Execution Time Comparison (512x512 Matrix)')
    ax1.set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax1.text(i, v/2, f"{v:.6f}s", ha='center', color='white', fontweight='bold')
    
    # TFLOPS comparison
    ax2.bar(labels, tflops, color=colors)
    ax2.set_title(f'Performance (TFLOPS) - Speedup: {speedup:.2f}x')
    ax2.set_ylabel('TFLOPS')
    for i, v in enumerate(tflops):
        ax2.text(i, v/2, f"{v:.2f}", ha='center', color='white', fontweight='bold')
    
    # Add individual runs as scatter points
    for i, t in enumerate(torch_times):
        ax1.scatter(0, t, color='black', alpha=0.5, s=30)
        ax2.scatter(0, (FLOP/t)*1e-12, color='black', alpha=0.5, s=30)
    
    for i, t in enumerate(custom_times):
        ax1.scatter(1, t, color='black', alpha=0.5, s=30)
        ax2.scatter(1, (FLOP/t)*1e-12, color='black', alpha=0.5, s=30)
    
    plt.tight_layout()
    plt.savefig('matmul_benchmark_512.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBenchmark Summary (512x512 Matrix):")
    print(f"PyTorch matmul: {avg_torch_time:.6f}s, {avg_torch_tflops:.2f} TFLOPS")
    print(f"Custom kernel: {avg_custom_time:.6f}s, {avg_custom_tflops:.2f} TFLOPS")
    print(f"Speedup: {speedup:.2f}x")

def main():
    print("Running benchmark for 512x512 matrices...\n")
    torch_times, custom_times = benchmark()
    create_visualization(torch_times, custom_times)

if __name__ == "__main__":
    main()