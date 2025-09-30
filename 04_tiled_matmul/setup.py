# In: setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='tiled_matmul',
            
            sources=[
                'src/matmul_tiled.cpp',
                'src/matmul_tiled_kernel.cu',
            ],
            include_dirs=['../common/include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)