Installation
############

Installation from GitHub
========================

To obtain the latest development version, clone
`the repository <https://github.com/timothyas/xesn>`_
and install it as follows::

    git clone https://github.com/timothyas/xesn.git
    cd xesn
    pip install -e .

Note that additional dependencies can be installed to run the unit test suite,::

    pip install -e .[test]
    pytest xesn/test/*.py

or to install `cupy <https://cupy.dev/>`_ to run on GPUs::

    pip install -e .[gpus]

Alternatively, the package can be installed via pip::

    pip install git+https://github.com/timothyas/xesn.git

Users are encourged to `fork <https://help.github.com/articles/fork-a-repo/>`_
the project and submit 
`issues <https://github.com/timothyas/xesn/issues>`_
and
`pull requests <https://github.com/timothyas/xesn/pulls>`_.

Running Example Notebooks or Building the Documentation Locally
===============================================================

Due to the way pandoc is installed via pip `as detailed here
<https://stackoverflow.com/a/71585691>`_
it is recommended to create an environment with conda in order to build the
documentation locally.
This is also recommended for running any of the example notebooks locally, since
there are a couple of additional dependencies required::

    conda env create -f docs/environment.yaml
    conda activate xesn
