import os
from setuptools import setup, find_packages

# Base dependencies for mytorch
INSTALL_REQUIRES = [
    "requests",
    "datasets<4.0.0",
    "wandb",
    "tqdm",
    "tiktoken",
    "safetensors",
    "cupy-cuda12x",
]

# Optional dependencies for mytorch[triton]
EXTRAS_REQUIRE = {
    "triton": [
        "torch>=2.0.0",
        "triton==3.5",
    ]
}

setup(
    name="mytorch",
    version="0.1.0",
    author="Priyam Mazumdar",
    description="A package for GPU-accelerated training with CuPy and optional Triton kernels",
    url="https://github.com/priyammaz/MyTorch",
    packages=find_packages(),
    python_requires=">=3.10,<3.14",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)