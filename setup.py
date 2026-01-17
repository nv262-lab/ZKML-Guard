"""Setup script for ZKML-Guard."""
from setuptools import setup, find_packages

setup(
    name="zkml-guard",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    author="ZKML-Guard Contributors",
    description="Verifiable Inference for Blind Signing Prevention",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zkml-guard/zkml-guard",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
