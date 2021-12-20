
import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_file, "r") as f:
        return f.read()


if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")


long_description = get_long_description()

setup(
    name="cwarp_rnnt",
    version="0.1.0",
    description="PyTorch bindings for CUDA-Warp RNN-Transducer in COMPACT layout.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cwarp_rnnt._C",
            sources=["cwarp_cuda.cpp", "cwarp_cuda_gather_kernel.cu",
                     "cwarp_cuda_rnnt_kernel.cu"],
            extra_compile_args={
                'nvcc': [
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
