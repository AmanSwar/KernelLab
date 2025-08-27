from setuptools import setup
from torch.utils.cpp_extension import BuildExtension , CUDAExtension

setup(
    name="matmul_kernel",
    ext_modules=[
        CUDAExtension(
            'matmul_kernel',
            sources=['python/binding.cpp' , 'src/regblock_gemm.cu'],
            include_dirs=['include']
        ),
        
    ],
    cmdclass={"build_ext" : BuildExtension}
)

