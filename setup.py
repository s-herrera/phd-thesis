from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "grex2.grex.features",
        ["src/grex2/grex/features.pyx"]
    )
]

setup(
    # to compile the .pyx files
    ext_modules=cythonize(extensions)
)