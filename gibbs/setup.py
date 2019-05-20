# Yibo Yang, 05/19/2019
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

import numpy

extensions = [
    Extension('disc_mrf_sampler',
              sources=['disc_mrf_sampler.pyx'],
              language='c++',
              include_dirs=[numpy.get_include()],  # for numpy.math.INFINITY
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              ),
]

setup(ext_modules=cythonize(extensions))
