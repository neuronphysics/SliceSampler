from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext
extra_compile_args = ['-std=c++11']
extra_link_args = ['-Wall']
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=[
        Extension("Slice_Sampler", 
                  sources=["Slice_Sampler.pyx"],
                  language="c++", 
                  libraries=["stdc++","gsl", "gslcblas"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)
    ],
gdb_debug=True)


