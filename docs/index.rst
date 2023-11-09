.. xesn documentation master file, created by
   sphinx-quickstart on Fri Oct  6 14:42:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xesn Documentation
==================

**xesn** is a python package for implementing Echo State Networks (ESNs), a
particular form of Recurrent Neural Network originally introduced by
:cite:t:`jaeger_echo_2001`.
The main purpose of the package is to enable ESNs for relatively large scale
weather and climate applications,
for example as by :cite:t:`smith_temporal_2023` and
:cite:t:`arcomano_machine_2020`.
The package is designed to strike the balance between simplicity and
flexibility, with a focus on implementing features that were shown to matter
most by :cite:t:`platt_systematic_2022`.

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
   methods
   example_esn_usage
   example_lazy_usage
   references
   api
   Source Code <https://github.com/timothyas/xesn/>

