#!/usr/bin/env python3

from setuptools import find_packages, setup
from setuptools.extension import Extension

INSTALL_REQUIRES = ['numpy', 'scipy']
TESTS_REQUIRE    = ['pytest >= 2.7.1']

import numpy
import os
#from Cython.Distutils import build_ext   # Extension for a c-file, build_ext for cython file
from Cython.Build import cythonize


#  cython stuff
#modules = ["fastnum", "hc_bcast", "cdf_smp", "ig_from_cdf_pkg", "conv_gau", "iwish"]
modules = ["iwish", "hc_bcast", "fastnum", "cdf_smp_1d", "cdf_smp_2d"]
tool_modules = ["compress_gz_pyx"]

#incdir = [get_python_inc(plat_specific=1), numpy.get_include(), "pyPG/include/RNG"]
incdir = [numpy.get_include()]
libdir = ['/usr/local/lib/gcc/6', '/usr/local/lib']
os.environ["CC"]  = "gcc-8"
os.environ["CXX"] = "gcc-8"

extra_compile_args = []
extra_link_args    = ["-lblas", "-lgsl"]
#####

extensions = []
for module in modules:
    extensions.append(Extension(module,
                                ["%s.pyx" % module],
                                #libraries = ['gsl', 'gslcblas'],
                                include_dirs=incdir,   #  include_dirs for Mac
                                library_dirs=libdir,
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args)  #  linker args
    )

tools_extensions = []
for module in tool_modules:
    tools_extensions.append(Extension(module,
                                      ["tools/%s.pyx" % module],
                                      #libraries = ['gsl', 'gslcblas'],
                                      include_dirs=incdir,   #  include_dirs for Mac
                                      library_dirs=libdir,
                                      extra_compile_args=extra_compile_args,
                                      extra_link_args=extra_link_args)  #  linker args
    )

setup(
    name='sumojam',
    version='0.1.0.dev0',
    license='MIT',
    description=(''),
    author='Kensuke Arai',
    author_email='kensuke.y.arai@gmail.com',
    url='https://github.com/AraiKensuke',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    scripts=[],
    ext_modules = cythonize(extensions)
)

setup(
    name='sumojam',
    version='0.1.0.dev0',
    license='MIT',
    description=(''),
    author='Kensuke Arai',
    author_email='kensuke.y.arai@gmail.com',
    url='https://github.com/AraiKensuke',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    scripts=[],
    ext_modules = cythonize(tools_extensions)
)

#
