import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#ROOT is the absolute path of this file.
ROOT = osp.dirname(osp.abspath(__file__))

#setup and distribute python packages
setup(
    name = 'dpvo',
    packages = find_packages(),
    ext_modules = [
        CUDAExtension('cuda_corr', sources=['dpvo/altcorr/correlation.cpp','dpvo/altcorr/correlation_kernel.cu'],
                      extra_compile_args={
                          'cxx':['-03'],
                          'nvcc':['-03'],
                      }),

    ]
)
