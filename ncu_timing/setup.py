from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='timing_kernel_pybind',
    ext_modules=[
        CUDAExtension(
            name='timing_kernel_pybind',
            sources=[os.path.join(current_dir, 'timing_kernel_pybind.cu')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_90']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

