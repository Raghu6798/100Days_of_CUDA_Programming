# In: setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='vector_add_cpp',
            
            sources=[
                'src/vector_add.cpp',
                'src/vector_add_kernel.cu',
            ],
            include_dirs=['../common/include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)