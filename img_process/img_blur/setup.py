from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_blur",
    ext_modules=[
        CUDAExtension(
            "cuda_blur_naive",
            ["img_process/img_blur/naive/blur_naive.cu"]
        ),
        CUDAExtension(
            "cuda_blur_optim",
            ["img_process/img_blur/optim/blur_optim.cu"]
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
