from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            # The name of the Python module we will import
            name='saxpy_cpp', 
            
            # The list of our new source files
            sources=[
                'src/saxpy.cpp',
                'src/saxpy_kernel.cu',
            ],
            # We still need the common utils for error checking if you use them
            include_dirs=['../common/include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)