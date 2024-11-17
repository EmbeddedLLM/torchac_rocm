import os
import subprocess
import sys
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension

def _is_hip() -> bool:
    return torch.version.hip is not None or os.environ.get('HIP_PLATFORM') == 'hcc'

def run_hipify():
    hipify_script = 'hipify.py'
    source_files = ['cal_cdf.cu', 'torchac_kernel_dec_new.cu', 'torchac_kernel_enc_new.cu']

    print([sys.executable, hipify_script, '-p', '.', '-o', 'hip_output', ' '.join(source_files)])

    subprocess.run([sys.executable, hipify_script, '-p', '.', '-o', 'hip_output'] + source_files)

is_hip = _is_hip()

if is_hip:
    print("ROCm environment detected. Running hipify...")
    run_hipify()
else:
    print("CUDA environment detected.")

source_files = ['main.cpp']
if is_hip:
    source_files.extend(['hip_output/cal_cdf.hip', 'hip_output/torchac_kernel_dec_new.hip', 'hip_output/torchac_kernel_enc_new.hip'])
else:
    source_files.extend(['cal_cdf.cu', 'torchac_kernel_dec_new.cu', 'torchac_kernel_enc_new.cu'])

extra_compile_args = {'cxx': ['-O3']}
define_macros = []

if is_hip:
    rocm_home = os.environ.get('ROCM_HOME', '/opt/rocm')
    hip_include = os.path.join(rocm_home, 'include')
    hipcub_include = os.path.join(rocm_home, 'include/hipcub')

    extra_compile_args['hip'] = ['-O3', f'-I{hip_include}', f'-I{hipcub_include}']
    define_macros.append(('__HIP_PLATFORM_HCC__', '1'))
else:
    extra_compile_args['nvcc'] = ['-O3']

setup(
    name = 'torchac_cuda',
    version = '0.2.5',
    description = 'GPU based arithmetic coding for LLM KV compression',
    author = 'Yihua Cheng',
    author_email = 'yihua98@uchicago.edu',
    include_package_data = True,
    ext_modules=[
        cpp_extension.CUDAExtension(
            'torchac_cuda', 
            source_files,
            extra_compile_args=extra_compile_args,
            include_dirs=['./include', hip_include, hipcub_include] if is_hip else ['./include'],
            define_macros=define_macros
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    install_requires = [
        "torch >= 2.1.0",
    ]
)