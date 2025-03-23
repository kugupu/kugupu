"""
kugupu
KUGUPUUU!
"""
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import versioneer
import os

DOCLINES = __doc__.split("\n")

lib_dirs = []
include_dirs = []
if 'CONDA_PREFIX' in os.environ:
    include_dirs.append(os.path.join(os.environ['CONDA_PREFIX'], 'include'))
if 'MAMBA_ROOT_PREFIX' in os.environ:
    include_dirs.append(
        os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'kgp', 'include')
    )
    lib_dirs.append(
        os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'kgp', 'lib')
    )

# Define Cython modules
extensions = [
    Extension(name='kugupu.time',
              sources=['src/kugupu/time.pyx'],
              extra_compile_args=[
                  '-std=c99', '-ffast-math', '-O3',
              ],
              define_macros=[('CYTHON_TRACE', '1')],
              ),
    Extension(name='kugupu._pyeht',
              sources=['src/kugupu/pyeht.pyx'],
              libraries=['yaehmop_eht', 'lapack', 'blas'],
              library_dirs=lib_dirs,
              include_dirs=include_dirs,
              extra_compile_args=[
                  '-std=c99', '-ffast-math', '-O3',
              ],
              )
]

setup(
    # Self-descriptive entries which should always be present
    name='kugupu',
    author='Micaela Matta and Richard J Gowers',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    packages=find_packages(),
    package_dir={'': 'src'},
    scripts=['bin/kugupu'],
    ext_modules=cythonize(extensions,
                          compiler_directives={'linetrace': True},
                          ),
    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    include_package_data=True,
    package_data={"kugupu": [
        "*.bz2",
        "*.hdf5",
        "*.dcd",
        "*.pdb",
        "*.yaml",
    ]},
)
