import vec_add_kernels
import torch



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


    # st = time.monotonic()
    # c = vec_add_kernels.naive_vec_add(a , b)

    # et = time.monotonic() - st
    # print(f"CUDA kernel execution time: {et:.6f} seconds")
    # checker(c)

    total_time_kernel = 0
    total_time_torch = 0
    for i in range(5):
        st_1 = time.monotonic()
        c_1 = vec_add_kernels.multielement_vec_add(a ,b)
        et_1 = time.monotonic() - st_1
        print(f"CUDA kernel execution time (multielement): {et_1:.6f} seconds")
        total_time_kernel += et_1

        st = time.monotonic()
        c = torch.add(a , b)
        et = time.monotonic() - st
        print(f"CUDA kernel execution time (pytorch): {et:.6f} seconds")
        total_time_torch += et

    print("torch : " , total_time_torch / 5)
    print("kernel : " , total_time_kernel / 5)
    print(c_1 == c)
if __name__ == "__main__":
    main()

    