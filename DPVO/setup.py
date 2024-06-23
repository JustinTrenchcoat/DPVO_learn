import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#ROOT is the absolute path of this file.
ROOT = osp.dirname(osp.abspath(__file__))

#setup and distribute python packages
#this file sets up three packages in total, the name for those packages are:
#cuda_corr, imported in altcorr folder by correlation.py
#cuda_ba, imported in fastba folder by ba.py
#lietorch_backends, imported in lietorch folder by group_ops.py
setup(
    name = 'dpvo',
    packages = find_packages(),
    ext_modules = [
        #creates three CUDAExtension packages
        CUDAExtension('cuda_corr', 
                      #the sources of package cuda_corr comes from the following two files.
                      sources=['dpvo/altcorr/correlation.cpp','dpvo/altcorr/correlation_kernel.cu'],
                      extra_compile_args={
                          #O3 is the optimization level, still have no idea what this is.
                          'cxx':['-O3'],
                          'nvcc':['-O3'],
                          }),
        CUDAExtension('cuda_ba',
                      sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu'],
                      extra_compile_args={
                          'cxx':['-O3']
                          }),
        CUDAExtension('lietorch_backends',
                      include_dirs=[
                          #combines the ROOT with th following to get two new paths
                          osp.join(ROOT, 'dpvo/lietorch/include'),
                          osp.join(ROOT,'thirdparty/eigen-3.4.0')],
                      sources=[
                              'dpvo/lietorch/src/lietorch.cpp',
                              'dpvo/lietrch/src/lietorch_gpu.cu',
                              'dpvo/lietorch/src/lietorch_cpu.cpp'],
                    extra_compile_args={'cxx':['-O3'],'nvcc':['-O3'],}),
    ],
    cmdclass={
        'build_ext':BuildExtension
        }
)
