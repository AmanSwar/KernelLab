import torch
import vec_add_kernels

a = torch.rand(size=(1000000,1) , device="cuda")
b = torch.rand(size=(1000000,1) , device="cuda")

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


    st = time.monotonic()
    c = vec_add_kernels.naive_vec_add(a , b)

    et = time.monotonic() - st
    print(f"CUDA kernel execution time: {et:.6f} seconds")
    checker(c)

if __name__ == "__main__":
    main()

    