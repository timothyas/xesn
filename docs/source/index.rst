.. xesn documentation master file, created by
   sphinx-quickstart on Fri Oct  6 14:42:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xesn Documentation
==================

**xesn** is a python package for implementing Echo State Networks (ESNs), a
particular form of Reservoir Computing originally discovered by
[Jaeger_2001]_.
The implementation makes use of 
`numpy <https://numpy.org/>`_ and
`scipy <https://scipy.org/>`_ for an efficient implementation on CPUs,
and `cupy <https://cupy.dev/>`_ for GPUs.


.. toctree::
   :maxdepth: 1

   installation
   notebooks/basic_usage
   notebooks/lazy_usage
   references
   api

