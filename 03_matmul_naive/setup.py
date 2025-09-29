# In: 03_matmul_naive/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name='matmul_cpp', # Use a simple, new name
            sources=[
                'matmul.cpp',
                'matmul_kernel.cu',
            ],
            include_dirs=['../common/include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)