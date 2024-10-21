import os
import sys
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    long_description = open('README.md').read()
except FileNotFoundError:
    long_description = 'CUDA implementation of BitNet for PyTorch'

def get_git_version():
    try:
        version = subprocess.check_output(['git', 'describe', '--tags']).decode().strip()
        return version
    except:
        return '0.0.1'

def find_cuda():
    try:
        return os.environ['CUDA_HOME']
    except KeyError:
        try:
            nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
            return os.path.dirname(os.path.dirname(nvcc))
        except:
            raise EnvironmentError("CUDA not found. Please set CUDA_HOME or add nvcc to your PATH.")

def get_cuda_version(cuda_home):
    version_file = os.path.join(cuda_home, 'version.txt')
    if os.path.isfile(version_file):
        with open(version_file, 'r') as f:
            version = f.read().strip().split()[-1]
        return version
    raise FileNotFoundError("Cannot find CUDA version file")

def get_cuda_arch_flags(cuda_version):
    major, minor = map(int, cuda_version.split('.')[:2])
    arch_flags = ['-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_75,code=sm_75']
    if major >= 11:
        arch_flags.append('-gencode=arch=compute_80,code=sm_80')
    if major >= 11 and minor >= 1:
        arch_flags.append('-gencode=arch=compute_86,code=sm_86')
    return arch_flags

version = get_git_version()
cuda_home = find_cuda()
cuda_version = get_cuda_version(cuda_home)
cuda_arch_flags = get_cuda_arch_flags(cuda_version)

ext_modules = [
    CUDAExtension('bitnet_cuda', [
        'src/bitnet_cuda.cpp',
        'src/bitnet_cuda_kernel.cu',
        'src/quantization_kernels.cu',
        'src/matmul_kernels.cu',
        'src/activation_kernels.cu',
        'src/memory_manager.cpp',
        'src/auto_tuner.cpp',
    ],
    include_dirs=[os.path.join(cuda_home, 'include')],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3'] + cuda_arch_flags
    })
]

if __name__ == '__main__':
    print(f"Setting up BitNet CUDA extension (version {version})")
    print(f"CUDA Home: {cuda_home}")
    print(f"CUDA Version: {cuda_version}")
    print(f"CUDA Architectures: {' '.join(cuda_arch_flags)}")

    try:
        setup(
            name='bitnet_cuda',
            version=version,
            author='Your Name',
            author_email='your.email@example.com',
            description='CUDA implementation of BitNet for PyTorch',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/yourusername/bitnet_cuda',
            packages=find_packages(),
            ext_modules=ext_modules,
            cmdclass={'build_ext': BuildExtension},
            install_requires=[
                'torch>=1.7.0',
            ],
            extras_require={
                'test': ['pytest'],
                'docs': ['sphinx', 'sphinx_rtd_theme'],
            },
            classifiers=[
                'Development Status :: 3 - Alpha',
                'Intended Audience :: Developers',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                'Programming Language :: Python :: 3.9',
            ],
            python_requires='>=3.6',
        )
    except Exception as e:
        print(f"Error during setup: {e}", file=sys.stderr)
        sys.exit(1)