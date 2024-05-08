
import os
from os.path import join
import yaml
import logging
import inspect
from contextlib import redirect_stdout
from copy import deepcopy
import re

import numpy as np
import xarray as xr
import dask.array as darray

from . import _use_cupy
from .cost import CostFunction, nrmse, psd
from .esn import ESN
from .io import from_zarr
from .lazyesn import LazyESN
from .optim import optimize
from .timer import Timer
from .utils import get_samples, update_esn_kwargs
from .xdata import XData

class Driver():
    """This is intended to automate :class:`ESN` and :class:`LazyESN` usage. The main methods to use are:

    - :meth:`run_training`: train readout weights

    - :meth:`run_macro_training`: use the `surrogate modeling toolbox <https://smt.readthedocs.io/>`_ to perform Bayesian optimization and learn macro-scale network parameters

    - :meth:`run_test`: test a trained network on a number of random samples from a test dataset

    Please see `this page of the documentation <https://xesn.readthedocs.io/en/latest/example_driver.html>`_ for examples of all of these, and an example configuration file.
    The experiments are configured with the parameter dict :attr:`config`.
    This can be created either by specifying the path to a yaml file or by explicitly passing the dict itself, see :meth:`set_config`.

    Args:
        config (str or dict): either a path to a yaml file or dict containing experiment parameters
        output_directory (str, optional): directory to save results and write logs to
    """

    __slots__ = (
        "config", "output_directory",
        "walltime", "localtime",
        "esn_name", "ESN",
        "logfile", "logname", "logger",
    )

    def __init__(self,
                 config,
                 output_directory=None):

        self._make_output_directory(output_directory)
        self._create_logger()
        self.set_config(config)

        # Look for ESN or LazyESN
        if "esn" in self.config.keys():
            self.ESN = ESN
        elif "lazyesn" in self.config.keys():
            self.ESN = LazyESN

        self.esn_name = self.ESN.__name__.lower()

        self.walltime = Timer(filename=self.logfile)
        self.localtime = Timer(filename=self.logfile)

        self.print_log(" --- Driver Initialized --- \n")
        self.print_log(self)


    def __str__(self):
        mystr = "Driver\n"+\
                f"    {'output_directory:':<28s}{self.output_directory}\n"+\
                f"    {'logfile:':<28s}{self.logfile}\n"
        return mystr


    def __repr__(self):
        return self.__str__()


    def run_training(self):
        """Perform :class:`ESN` or :class:`LazyESN` training, learn the readout matrix weights.
        Resulting readout weights are stored in the :attr:`output_directory` in the zarr store ``esn-weights.zarr`` or ``lazyesn-weights.zarr``, depending on which model is used. See :meth:`ESN.to_xds` for what is stored in this dataset.

        Required Parameter Sections:
            xdata: all options are used to create an :class:`XData` object. If ``subsampling`` is provided, it must have slicing options labelled "training".
            esn or lazyesn: all options are used to create a :class:`ESN` or :class:`LazyESN` object, depending on the name of the section, case does not matter
            training: all options are passed to :meth:`ESN.train` or :meth:`LazyESN.train`

        Example Config YAML File:

            Highlighted regions are used by this routine, other options are ignored.

            .. literalinclude:: ../config.yaml
                :language: yaml
                :emphasize-lines: 1-8, 12-45
        """

        self.walltime.start("Starting Training")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="training")

        if "lazy" not in self.esn_name:
            xda = xda.load()
        self.localtime.stop()

        # setup ESN
        self.localtime.start(f"Building {self.esn_name}")
        esn = self.ESN(**self.config[self.esn_name])
        esn.build()
        self.print_log(str(esn))
        self.localtime.stop()

        self.localtime.start(f"Training {self.esn_name}")
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                esn.train(xda, **self.config["training"])
        self.localtime.stop()

        self.localtime.start(f"Storing {self.esn_name} Weights")
        ds = esn.to_xds()
        if _use_cupy:
            ds = ds.as_numpy()
        ds.to_zarr(join(self.output_directory, f"{self.esn_name}-weights.zarr"), mode="w")
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


    def run_macro_training(self):
        """Perform Bayesian optimization on macro-scale ESN parameters using surrogate modeling toolbox.
        Resulting optimized parameters are written to ``config-optim.yaml`` and to the log file ``stdout.log`` in the :attr:`output_directory`

        Required Parameter Sections:
            xdata: all options are used to create an :class:`XData` object. If ``subsampling`` is provided, it must have slicing options labelled "training" and "macro_training".
            esn or lazyesn: all options are used to create a :class:`ESN` or :class:`LazyESN` object, depending on the name of the section, case does not matter
            training: all options are passed to :meth:`ESN.train` or :meth:`LazyESN.train`
            macro_training: with the following subsections

                - "parameters" (required): with key/value pairs as the parameters to be optimized, and their bounds as values

                - "transformations" (optional): with any desired transformations on the input variables pre-optimization, see :func:`xesn.optim.transform` for example

                - "forecast" (required): with options for sample forecasts to optimize macro parameters with, see :func:`get_samples` for a list of parameters (other than xda)

                - "ego" (required): with parameters except for evaluation/cost function (which is defined by a :class:`CostFunction`) and surrogate (assumed to be ``smt.surrogate_models.KRG``) as passed to `smt.applications.EGO <https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html#options>`_

                - "cost_terms" (optional): forms the cost function defined in :class:`CostFunction` by determining the weights for the NRMSE and PSD\_NRMSE cost terms (default: ``{"nrmse": 1}``)

                - "cost_upper_bound" (optional): remove sensitivity to values larger than this number by setting all ``numpy.inf``, ``numpy.nan``, and any value greater than this threshold to this number (default: ``1.e9``)

        Example Config YAML File:

            Highlighted regions are used by this method, other options are ignored.

            .. literalinclude:: ../config.yaml
                :language: yaml
                :emphasize-lines: 1-9, 12-45, 59-85
        """

        self.walltime.start("Starting Macro Training")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        # First macro data sets
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="macro_training")

        if "lazy" not in self.esn_name:
            xda = xda.load()

        macro_data, indices = get_samples(xda=xda, **self.config["macro_training"]["forecast"])
        if "sample_indices" not in self.config["macro_training"]["forecast"]:
            self.overwrite_config({"macro_training": {"forecast": {"sample_indices": indices}}})

        # Now training data
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="training")
        if "lazy" not in self.esn_name:
            xda = xda.load()
        self.localtime.stop()

        # create cost function
        self.localtime.start("Setting up cost function")
        cf = CostFunction(self.ESN, xda, macro_data, self.config)
        self.localtime.stop()

        # optimize
        self.localtime.start("Starting Bayesian Optimization")
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                p_opt = optimize(cf, **self.config["macro_training"]["ego"])
        self.localtime.stop()

        config_optim = {self.esn_name: update_esn_kwargs(p_opt)}
        self.overwrite_config(config_optim)
        outname = os.path.join(self.output_directory, "config-optim.yaml")
        with open(outname, "w") as f:
            yaml.dump(self.config, stream=f)

        self.print_log(f"Optimal configuration written to {outname}")

        self.walltime.stop()


    def run_test(self):
        """Make test predictions using a pre-trained ESN.
        Results are stored in a zarr store in the :attr:`output_directory` as ``test-results.zarr``.

        Required Parameter Sections:
            xdata: all options are used to create an :class:`XData` object. If ``subsampling`` is provided, it must have slicing options labelled "testing".
            esn_weights: all options are passed to :func:`from_zarr` to create the :class:`ESN` or :class:`LazyESN` object
            testing: all options passed to :func:`get_samples`, except the required "xda" parameter for that function.  In this section the user can also optionally provide the ``"cost_terms"`` dict similar to what is used in :meth:`run_macro_training`. If this is included, it adds NRMSE and/or PSD_NRMSE metrics to the ``test-results.zarr`` store, based on what is included in this dictionary. Note that the values in ``"cost_terms"`` in this section are ignored - weighting can be done offline.

        Example Config YAML File:

            Highlighted regions are used by this method, other options are ignored.

            .. literalinclude:: ../config.yaml
                :language: yaml
                :emphasize-lines: 1-7, 10-14, 47-57
        """

        self.walltime.start("Starting Testing")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="testing")

        if "lazy" not in self.esn_name:
            xda = xda.load()
        self.localtime.stop()

        # first get cost_terms, so we can unpack config conveniently in get_samples
        cfg = self.config.copy()
        cost_terms = cfg["testing"].pop("cost_terms", {})

        # pull samples from data
        self.localtime.start("Get Test Samples")
        test_data, indices = get_samples(xda=xda, **cfg["testing"])
        if "sample_indices" not in self.config["testing"]:
            self.overwrite_config({"testing": {"sample_indices": indices}})
        self.localtime.stop()

        # setup ESN from zarr
        self.localtime.start("Read ESN Zarr Store")
        esn = from_zarr(**self.config["esn_weights"])
        self.localtime.stop()

        # make predictions
        self.localtime.start("Make Test Predictions")
        zpath = join(self.output_directory, "test-results.zarr")
        for i, tester in enumerate(test_data):
            xds = esn.test(
                tester,
                n_steps=self.config["testing"]["n_steps"],
                n_spinup=self.config["testing"]["n_spinup"]
            )
            xds["prediction"] = data.normalize_inverse(xds["prediction"], keep_attrs=True)
            xds["truth"] = data.normalize_inverse(xds["truth"], keep_attrs=True)
            xds.attrs["initial_condition_index"] = indices[i]

            # evaluate cost, if applicable
            if "nrmse" in cost_terms:
                xds["nrmse"] = nrmse(xds, drop_time=False)

            if "psd_nrmse" in cost_terms:
                xds["psd_truth"] = psd(xds["truth"])
                xds["psd_prediction"] = psd(xds["prediction"])
                xds["psd_nrmse"] = nrmse({
                    "truth": xds["psd_truth"],
                    "prediction": xds["psd_prediction"],
                    }, drop_time=False)

            # Make container and store this sample
            if i == 0:
                self._make_container(zpath, xds, n_samples=len(test_data))

            xds = xds.expand_dims({"sample": [i]})
            region = {d: slice(None, None) for d in xds.dims}
            region["sample"] = slice(i, i+1)
            if _use_cupy:
                xds = xds.as_numpy()
            xds.to_zarr(zpath, region=region)

        self.localtime.stop()

        self.walltime.stop()


    def _make_container(self, zstore_path, xds, n_samples, **kwargs):
        """Create a container zarr store with empty values for the test results"""

        cds = xr.Dataset()
        cds["sample"] = xr.DataArray(
            np.arange(n_samples),
            coords={"sample": np.arange(n_samples)},
            dims="sample",
            attrs={"description": "test sample index"},
        )
        for d in xds.dims:
            cds[d] = xds[d]

        for key in xds.data_vars:
            dims = ("sample",) + xds[key].dims
            chunks = xds[key].data.chunksize if isinstance(xds[key].data, darray.Array) else \
                     tuple(-1 for _ in xds[key].dims)
            chunks = (1,) + chunks
            shape = (n_samples,) + xds[key].shape

            cds[key] = xr.DataArray(
                data=darray.zeros(
                    shape=shape,
                    chunks=chunks,
                    dtype=xds[key].dtype,
                ),
                coords={"sample": cds["sample"], **{d: xds[d] for d in xds[key].dims}},
                dims=dims,
                attrs=xds[key].attrs.copy(),
            )

        cds.to_zarr(zstore_path, compute=False, **kwargs)


    def _make_output_directory(self, out_dir):
        """Make provided output directory. If none given, make a unique directory:
        output-driver-XX, where XX is 00->99

        Args:
            out_dir (str or None): path to dump output, or None for default

        Sets Attributes:
            out_dir (str): path to created output directory
        """
        if out_dir is None:

            # make a unique default directory
            i=0
            out_dir = f"output-driver-{i:02d}"
            while os.path.isdir(out_dir):
                if i>99:
                    raise ValueError("Hit max number of default output directories...")
                out_dir = f"output-driver-{i:02d}"
                i = i+1
            os.makedirs(out_dir)

        elif not os.path.isdir(out_dir):
            print("Creating directory for output: ",out_dir)
            os.makedirs(out_dir)

        self.output_directory = out_dir


    def _create_logger(self):
        """Create a logfile and logger for writing all text output to

        Sets Attributes:
            logfile (str): path to logfile: ouput_directory / stdout.log
            logname (str): name of logger
            logger (:obj:`logging.Logger`): used to write to file
        """

        # create a log file
        self.logfile = os.path.join(self.output_directory, 'stdout.log')

        # create a logger
        self.logname = f'driver_logger'
        self.logger = logging.getLogger(self.logname)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.logfile)
        fmt = logging.Formatter(style='{')
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)


    def set_config(self, config):
        """Read the nested parameter dictionary or take it directly, and write a copy for
        reference in the :attr:`output_directory`.

        Args:
            config (str or dict): filename (path) to the configuration yaml file, or nested dictionary with parameters

        Sets Attribute:
            config (dict): with a big nested dictionary with all parameters
        """

        if isinstance(config, str):
            config = self.load(config)

        elif not isinstance(config, dict):
            raise TypeError(f"Driver.set_config: Unrecognized type for experiment config, must be either yaml filename (str) or a dictionary with parameter values")

        # make the section names lower case
        lconfig = {}
        for key in config.keys():
            lconfig[key.lower()] = config[key]

        self._check_config_options(lconfig)
        self.config = lconfig

        outname = os.path.join(self.output_directory, "config.yaml")
        with open(outname, "w") as f:
            yaml.dump(self.config, stream=f)


    def overwrite_config(self, new_config):
        """Overwrite specific parameters with the values in the nested dict new_config.

        Args:
            new_config (dict): nested dictionary with values to overwrite object's parameters with

        Sets Attribute:
            config (dict): with the nested dictionary based on the input config file

        Example:
            >>> driver = Driver("config.yaml")
            >>> print(driver.config)
            {'xdata': {'dimensions': ['x', 'time'],
             'zstore_path': 'lorenz96-12d.zarr', ...},
             'esn': {'n_input': 12,
             'n_output': 12,
             'n_reservoir': 1000, ...}}
            >>> new_config = {'esn':{'n_reservoir':2000}, 'xdata': {'zstore_path': 'new_data.zarr'}}
            >>>
            >>> driver.overwrite_params(new_config)
            >>> print(driver.config)
            {'xdata': {'dimensions': ['x', 'time'],
             'zstore_path': 'new_data.zarr', ...},
             'esn': {'n_input': 12,
             'n_output': 12,
             'n_reservoir': 2000, ...}}
        """

        config = self.config.copy()
        for section, this_dict in new_config.items():
            for key, val in this_dict.items():
                s = section.lower()
                if not isinstance(val, dict):
                    self.print_log(f"Driver.overwrite_config: Overwriting driver.config['{s}']['{key}'] with {val}")
                    if s in config:
                        config[s][key] = val
                    else:
                        config[s] = {key: val}
                else:
                    for k2, v2 in val.items():
                        self.print_log(f"Driver.overwrite_config: Overwriting driver.config['{s}']['{key}']['{k2}'] with {v2}")
                        if s in config:
                            if key in config[s]:
                                config[s][key][k2] = v2
                            else:
                                config[s][key] = {k2: v2}
                        else:
                            config[s] = {key: {k2: v2}}


        # Overwrite our copy of config.yaml in output_dir and reset attr
        self.set_config(config)


    def print_log(self, *args, **kwargs):
        """Print to log file as specified in :attr:`logname`"""
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                print(*args, **kwargs)


    @staticmethod
    def load(fname):
        """An extension of :func:`yaml.safe_load` that recognizes 1e9 as float not string
        (i.e., don't require the 1.0 or the sign +9).

        Thanks to <https://stackoverflow.com/a/30462009>.

        Args:
            fname (str): path to yaml file

        Returns:
            config (dict): with the contents of the yaml file
        """


        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(fname, "r") as f:
            config = yaml.load(f, Loader=loader)
        return config


    def _check_config_options(self, config):
        """Make sure we recognize each configuration section name, and each option name.
        No type or value checking

        Args:
            config (dict): the big nested options dictionary
        """

        # Check sections
        expected = {
                "xdata": XData,
                "esn": ESN,
                "lazyesn": LazyESN,
                "training": LazyESN.train,
                "macro_training": None,
                "testing": None,
                "esn_weights": None}
        bad_sections = []
        for key in config.keys():
            try:
                assert key in expected.keys()
            except:
                bad_sections.append(key)

        if len(bad_sections)>0:
            raise KeyError(f"Driver._check_config_options: unrecognized config section(s): {bad_sections}")

        # Check options in each section
        for section in config.keys():
            Func = expected[section]
            if Func is not None:
                kw, *_ = inspect.getfullargspec(Func)
                for key in config[section].keys():
                    try:
                        assert key in kw
                    except:
                        raise KeyError(f"Driver._check_config_options: unrecognized option {key} in section {section}")
