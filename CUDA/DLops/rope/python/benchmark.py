from rope_cuda import rope_apply_cuda

import torch
import time


print(type(rope_apply_cuda))



def apply_rope(
        x : torch.Tensor, 
        cos : torch.Tensor, 
        sin : torch.Tensor
    ):
    batch_size , num_heads , seq_len , head_dim = x.shape

    assert head_dim % 2 == 0 , "Head dim is not divisible by 2"

    x1 = x[... , :head_dim // 2]
    x2 = x[... , head_dim// 2 : ]

    cos = cos[:seq_len , :].unsqueeze(0).unsqueeze(0) #-> (1 , 1 , seq_len , head_dim)
    sin = sin[:seq_len , :].unsqueeze(0).unsqueeze(0) #-> (1 , 1 , seq_len , head_dim)

    rotated = torch.cat((-x2 , x1) , dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


bs = 1
n_heads = 16
seq_len = 1024
head_dim = 128

device = torch.device("cuda")


x = torch.rand(size=(bs , n_heads , seq_len , head_dim) , dtype=torch.float16).to(device)
cos = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)
sin = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)

st = time.monotonic()
for _ in range(100):
  out = apply_rope(x , cos , sin)

et = time.monotonic() - st


st1 = time.monotonic()
for _ in range(100):
  cuda_out = rope_apply_cuda(x , cos , sin)
et1 = time.monotonic() - st1

print(f"torch time : {et}")
print(f"cuda time : {et1}")

