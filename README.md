# xesn

[![codecov](https://codecov.io/gh/timothyas/xesn/graph/badge.svg?token=X1Z9BZB5XS)](https://codecov.io/gh/timothyas/xesn)
[![Documentation Status](https://readthedocs.org/projects/xesn/badge/?version=latest)](https://xesn.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xesn.svg)](https://anaconda.org/conda-forge/xesn)
[![PyPI version](https://badge.fury.io/py/xesn.svg)](https://badge.fury.io/py/xesn)

Echo State Networks powered by
[xarray](https://docs.xarray.dev/en/stable/)
and
[dask](https://www.dask.org/).

## Description

**xesn** is a python package for implementing Echo State Networks (ESNs), a
particular form of Recurrent Neural Network originally introduced by
[Jaeger (2001)](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf).
The main purpose of the package is to enable ESNs for relatively large scale
weather and climate applications,
for example as by [Smith et al., (2023)](https://arxiv.org/abs/2305.00100)
and [Arcomano et al., (2020)](https://doi.org/10.1029/2020GL087776).
The package is designed to strike the balance between simplicity and
flexibility, with a focus on implementing features that were shown to matter
most by [Platt et al., (2022)](https://doi.org/10.1016/j.neunet.2022.06.025).

xesn uses [xarray](https://docs.xarray.dev/en/stable/)
to handle multi-dimensional data, relying on
[dask](https://www.dask.org/) for parallelization and
to handle datasets/networks that are too large for a single compute node.
At its core, xesn uses
[numpy](https://numpy.org/)
and [cupy](https://cupy.dev/)
for efficient CPU and GPU deployment.

## Installation

Installation from
[conda-forge](https://anaconda.org/conda-forge/xesn)

```shell
conda install -c conda-forge xesn
```

Installation from pip

```shell
pip install xesn
```

Installation from source

```shell
git clone https://github.com/timothyas/xesn.git
cd xesn
pip install -e .
```

Note that additional dependencies can be installed to run the unit test suite::

```shell
pip install -e .[test]
pytest xesn/test/*.py
```

## Getting Started

To learn how to use xesn, check out the
[documentation here](https://xesn.readthedocs.io/en/latest/index.html)

## Get in touch

Report bugs, suggest features, or view the source code
[on GitHub](https://github.com/timothyas/xesn).

## License and Copyright

xesn is licensed under version 3 of the GNU Lesser General Public License.

Development occurs on GitHub at <https://github.com/timothyas/xesn>.
