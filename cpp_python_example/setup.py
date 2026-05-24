from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "calculator",
        ["calculator.cpp", "bindings.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name="calculator",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10.0'],
    python_requires=">=3.6",
) 