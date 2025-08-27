# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension , CUDAExtension

# setup(
#     name="vec_add_kernels",
#     ext_modules=[
#         CUDAExtension(
#             'vec_add_kernels',
#             sources=['python/binding.cpp' , 'src/naive_vecadd.cu' , "src/multielement_vecadd.cu"],
#             include_dirs=['include']
#         ),
#     ],
#     cmdclass={"build_ext" : BuildExtension}
# )


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vec_add_kernels',
    ext_modules=[
        CUDAExtension(
            'vec_add_kernels',
            sources=['binding.cpp', '../src/naive_vecadd.cu', '../src/multielement_vecadd.cu'],
            include_dirs=['../include']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)