# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:11:58 2019

@author: ryanb
"""

import sys
from extra_vectorisation_openmp_V5 import main

if int(len(sys.argv)) == 3:
    main(int(sys.argv[1]),int(sys.argv[2]))
else:
    print("Usage: {} <ITERATIONS> <THREADS>".format(sys.argv[0]))