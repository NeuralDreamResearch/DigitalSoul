from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

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
print("Found packages:", packages)  # Add this line
setup(
    name='DigitalSoul',  # Name of your overall project
    version='1.0.0',
    description='My DigitalSoul project',
    packages=packages,
    ext_modules=[dscpp_ext],
    cmdclass={'build_ext': build_ext}  # Ensure build_ext is used for compilation
)

