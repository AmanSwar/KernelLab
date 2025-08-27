from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension , BuildExtension



setup(
    name="greyscale_kernel",
    ext_modules=[
        CUDAExtension(
            'greyscale_kernel',
            sources=['python/binding.cpp' , 'src/naive_greyscale.cu'],
            include_dirs=['include']
        ),
        
    ],
    cmdclass={"build_ext" : BuildExtension}
)

