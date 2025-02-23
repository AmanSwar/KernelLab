import torch
import cv2
import numpy as np
import time
import cuda_blur_naive
import cuda_blur_optim


image = cv2.imread("img_process/test_img/test_img.jpg" ,cv2.IMREAD_COLOR_RGB).astype(np.float32)
img_tensor = torch.tensor(image , device="cuda").float()

#----------------NAIVE---------------------------

st_kernel = time.time()

blur_tensor_naive = cuda_blur_naive.blur(img_tensor)

end_kernel = time.time() - st_kernel
print(end_kernel)

# ------------------OPTIM-------------------------

st_kernel = time.time()

blur_tensor_naive = cuda_blur_optim.blur(img_tensor)

end_kernel = time.time() - st_kernel
print(end_kernel)

# ------------------openCV--------------------------

st_cv2 = time.time()
blur_cv2 = cv2.blur(image , ksize=(3,3))
end_cv2 = time.time() - st_cv2
print(end_cv2)


print(f"{(end_cv2 / end_kernel):.3f} ")