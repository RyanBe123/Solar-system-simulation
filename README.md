# Solar-system-simulation
Gravity simulation using parallel computer on a `Blue Crystal Phase 3' supercomputer.

Newtonian Gravity was used to study the motion of bodies within a solar system. The long range nature of gravity nature requires the force felt on each body from every other body and so quickly becomes computationally expensive. To minimise the computational time, different optimisation techniques were employed.

The single_core.py programme contains the initial programme of the gravity simulator using `SUVAT' equations and Newtonian mechanics.
This was initially optimised by removing loops and vectorising with NumPy and can be seen in vectorised.py
Further to this the programme was 'Cythonised'.  The files for this optimisation can be found under cdef_vector_run.py, cdef_vector_setup.py and cdef_vector_cython.pyx.

The programmes were then parallelised using two libraries MPI and OpenMP.
MPI - MPI.py
OpenMP - run_extra_vectorisation_openmp_V5.py, setup_extra_vectorisation_openmp_V5.py and extra_vectorisation_openmp_V5.pyx

Plots of the timings of the different programmes is found in plots.py.

A report is also included discussing the background, theory and findings of the project.
