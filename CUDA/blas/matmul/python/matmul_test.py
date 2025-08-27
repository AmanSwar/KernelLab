import torch
import matmul_kernel
import time



SIZE = 512
FLOP = 2 * SIZE * SIZE * SIZE
mata = torch.rand(size=(SIZE , SIZE) , device="cuda")
matb = torch.rand(size=(SIZE , SIZE) , device="cuda")

def main():
    st_2 = time.monotonic()
    matd = torch.matmul(mata , matb)
    et_2 = time.monotonic() - st_2
    print("torch matmul:")
    print(et_2)
    print((FLOP/et_2)*1e-12)
    
    print("\n")

    st_1 = time.monotonic()
    matc = matmul_kernel.matmul_kernel(mata , matb)
    et_1 = time.monotonic() - st_1
    print("custom")
    print(et_1)
    print((FLOP/et_1)*1e-12)

    

    

if __name__ == "__main__":

    main()