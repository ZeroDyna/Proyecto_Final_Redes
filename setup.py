from setuptools import setup, Extension
import pybind11

### EL SETUP >:D

ext_modules = [
    Extension(
        "ucsp",
        ["bindings.cpp", "ucsp.cpp"],
        include_dirs=[pybind11.get_include()],
        libraries=["z"],
        extra_compile_args=["-std=c++17"],
        language="c++",
    ),
]

setup(
    name="ucsp",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.10.0"],
    python_requires=">=3.6",
)
