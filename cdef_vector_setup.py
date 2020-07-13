# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:47:59 2019

@author: ryanb
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name="cdef_vector_cython.pyx",
      ext_modules=cythonize("cdef_vector_cython.pyx"))