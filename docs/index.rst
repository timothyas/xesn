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
`dask <https://www.dask.org/>`_ for parallelization and
to handle datasets/networks that are too large for a single compute node.
At its core, xesn uses
`numpy <https://numpy.org/>`_ 
and `cupy <https://cupy.dev/>`_ 
for efficient CPU and GPU deployment.



.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   methods
   gpus

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Using the Standard ESN <example_esn_usage>
   Using the Parallel ESN <example_lazy_usage>
   Macro Parameter Optimization <example_macro_training>
   Writing Less Code with the Driver Class <example_driver>

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing
   support

.. toctree::
   :maxdepth: 1
   :caption: References

   references
   api
