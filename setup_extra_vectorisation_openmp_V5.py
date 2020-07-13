# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:47:43 2019

@author: ryanb
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
ext_modules = [Extension("extra_vectorisation_openmp_V5",
                        ["extra_vectorisation_openmp_V5.pyx"],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],)]
setup(name="extra_vectorisation_openmp_V5",
ext_modules=cythonize(ext_modules))
