from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 1. C++/Pybind11 Extension
dscpp_ext = Pybind11Extension(
    'DigitalSoul.dscpp',
    sources=['DigitalSoul/include/digital_soul_bindings.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    cxx_std=17 
)

# 2. Find Pure Python Packages
packages = find_packages()  # Automatically find 'DigitalSoul'
setup(
    name='DigitalSoul', 
    version='1.1.5',
    description='Unified Compute Platform - CPU, GPU, FPGA, Quantum Computing',
    packages=packages,
    ext_modules=[dscpp_ext],
    cmdclass={'build_ext': build_ext}, 
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls = {
    'Homepage': 'https://github.com/NeuralDreamResearch/DigitalSoul',
    'Documentation': 'https://github.com/NeuralDreamResearch/DigitalSoul', 
    'Source Code': 'https://github.com/NeuralDreamResearch/DigitalSoul',
    'Bug Tracker' : 'https://github.com/NeuralDreamResearch/DigitalSoul/issues'
	},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pybind11']
)

