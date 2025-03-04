import greyscale_kernel
import torch

image = torch.rand(size=(32 , 3 , 224 , 224) , device="cuda")

batch_size , channels , height , width = image.size()
image = image.permute(0, 2, 3, 1).contiguous().view(-1, channels)


out = greyscale_kernel.greyscale(image)

print(out)
