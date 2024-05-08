Installation
############

Installation from conda forge
=============================

xesn can be installed from conda-forge::

    conda install -c conda-forge xesn


Installation from pip
=====================

Alternatively, xesn can be installed using pip::

    pip install xesn


Installation from GitHub
========================

To obtain the latest development version, clone
`the repository <https://github.com/timothyas/xesn>`_
and install it as follows::

    git clone https://github.com/timothyas/xesn.git
    cd xesn
    pip install -e .

Note that additional dependencies can be installed to run the unit test suite
and ensure everything is working properly::

    pip install -e .[test]
    pytest xesn/test/*.py

Users are encourged to `fork <https://help.github.com/articles/fork-a-repo/>`_
the project and submit 
`issues <https://github.com/timothyas/xesn/issues>`_
and
`pull requests <https://github.com/timothyas/xesn/pulls>`_.

Running Example Notebooks or Building the Documentation Locally
===============================================================

Due to the way pandoc is installed via pip `as detailed here
<https://stackoverflow.com/a/71585691>`_
it is recommended to create an environment with conda
using
`this environment file
<https://github.com/timothyas/xesn/blob/main/ci/environment.yaml>`_
in order to build the documentation locally.
This is also recommended for running any of the example notebooks locally, since
there are a couple of additional dependencies required::

    conda env create -f docs/environment.yaml
    conda activate xesn

Using GPUs
==========

In order to use this package on GPUs, please install cupy and cupy-xarray as well, following
their installation instructions
`here <https://docs.cupy.dev/en/stable/install.html>`_
and 
`there <https://cupy-xarray.readthedocs.io/>`_
An appropriate conda environment can also be created by uncommenting the bottom 3 lines of
`this environment file
<https://github.com/timothyas/xesn/blob/main/ci/environment.yaml>`_
and creating the environment as follows::

    conda env create -f docs/environment.yaml
    conda activate xesn

Additionally, it may be necessary to require ``xarray<2024.02.0`` until
`this issue <https://github.com/xarray-contrib/cupy-xarray/issues/41>`_
is resolved.
