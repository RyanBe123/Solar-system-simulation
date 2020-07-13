# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:50:37 2019

@author: ryanb
"""

import sys
from cdef_vector_cython import main

if int(len(sys.argv)) == 2:
    main(int(sys.argv[1]))
else:
    print("Usage: {} <ITERATIONS>".format(sys.argv[0]))