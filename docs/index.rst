.. xesn documentation master file, created by
   sphinx-quickstart on Fri Oct  6 14:42:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xesn Documentation
==================

**xesn** is a python package for implementing Echo State Networks (ESNs), a
particular form of Recurrent Neural Network originally discovered by
[Jaeger_2001]_.
The main purpose of the package is to enable ESNs for relatively large scale
weather and climate applications,
for example as in [Smith_et_al_2023]_ and [Arcomano_et_al_2020]_.
The package is designed to strike the balance between simplicity and
flexibility, with a focus on implementing features that were shown to matter
most by [Platt_et_al_2022]_.

xesn uses `xarray <https://docs.xarray.dev/en/stable/>`_ 
to handle multi-dimensional data, relying on
`dask <https://www.dask.org/>`_ for parallelization.
At its core, xesn uses
`numpy <https://numpy.org/>`_ 
and `cupy <https://cupy.dev/>`_ 
for efficient CPU and GPU deployment.


.. toctree::
   :maxdepth: 1

   installation
   basic_usage
   lazy_usage
   references
   api

