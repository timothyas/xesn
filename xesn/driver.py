
import os
from os.path import join
import yaml
import logging
import inspect
from contextlib import redirect_stdout
import re

import numpy as np

from .cost import CostFunction
from .esn import ESN
from .io import from_zarr
from .lazyesn import LazyESN
from .optim import optimize
from .timer import Timer
from .xdata import XData

class Driver():
    """This is intended to automate the ESN usage. The main methods to use are:
        - :meth:`run_micro_calibration`: train ESN readout weights
        - :meth:`run_test`: test a trained ESN on a number of random samples from a test dataset

    The experiments are configured with a parameter dict, that can be created either
    with a yaml file or by explicitly passing the dict itself.

    Args:
        config (str or dict): either a path to a yaml file or dict containing experiment parameters
        output_directory (str, optional): directory to save results and write logs to
    """
    name                    = "driver"
    config                  = None
    output_directory        = None
    walltime                = None
    localtime               = None

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


    def run_micro_calibration(self):
        """Perform ESN training, learn the readout matrix weights.

        Required Parameter Sections:
            - "xdata" with options passed to :meth:`XData`, and expected time slices for "training"
            - "esn" or "lazyesn" with options passed to :meth:`ESN` or :meth:`LazyESN`
            - "training" with options passed to :meth:`ESN.train` or :meth:`LazyESN.train`
        """

        self.walltime.start("Starting Micro Calibration")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="training")
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
        ds.to_zarr(join(self.output_directory, f"{self.esn_name}-weights.zarr"), mode="w")
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


    def run_macro_calibration(self):
        """Perform Bayesian optimization on macro-scale ESN parameters using surrogate modeling toolbox.

        Required Parameter Sections:
            - "xdata" with options passed to :meth:`XData`, and expected time slices "macro_training" and "training"
            - "esn" or "lazyesn" with options passed to :meth:`ESN` or :meth:`LazyESN`
            - "training" with options passed to :meth:`ESN.train` or :meth:`LazyESN.train`
            - "macro_training" with a variety of subsections:
                - "parameters" (required) with key/value pairs as the parameters to be optimized, and their bounds as values
                - "transformations" (optional) with any desired transformations on the input variables pre-optimization, see :func:`xesn.optim.transform` for example
                - "forecast" with options for sample forecasts to optimize macro parameters with, see :meth:`get_samples` for a list of parameters (other than xda)
                - "ego" with parameters except for evaluation/cost function (which is defined by a :class:`CostFunction`) and surrogate (assumed to be ``smt.surrogate_models.KRG``) as passed to `smt.applications.EGO <https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html#options>`_
        """

        self.walltime.start("Starting Macro Calibration")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        # First macro data sets
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="macro_training")

        macro_data, indices = self.get_samples(xda=xda, **self.config["macro_training"]["forecast"])
        if "sample_indices" not in self.config["macro_training"]["forecast"]:
            self.overwrite_config({"macro_training": {"forecast": {"sample_indices": indices}}})

        # Now training data
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="training")
        self.localtime.stop()

        # create cost function
        self.localtime.start("Setting up cost function")
        cf = CostFunction(self.ESN, xda, macro_data, self.config)
        self.localtime.stop()

        # optimize
        self.localtime.start("Starting Bayesian Optimization")
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                p_opt = optimize(self.config["macro_training"]["parameters"],
                                 self.config["macro_training"]["transformations"],
                                 cf,
                                 **self.config["macro_training"]["ego"])
        self.localtime.stop()

        config_optim = self.config.copy()
        config_optim[self.esn_name].update(config_optim)
        outname = os.path.join(self.output_directory, "config-optim.yaml")
        with open(outname, "w") as f:
            yaml.dump(config_optim, stream=f)

        self.print_log(f"Optimal configuration written to {outname}")

        self.walltime.stop()


    def run_test(self):
        """Make test predictions using a pre-trained ESN.

        Required Parameter Sections:
            - "xdata" with options passed to :meth:`XData`
            - "esn_weights" with options passed to :func:`from_zarr`
            - "testing" with options passed to :meth:`get_samples`, except "xda"
        """

        self.walltime.start("Starting Testing")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                xda = data.setup(mode="testing")
        self.localtime.stop()

        # pull samples from data
        self.localtime.start("Get Test Samples")
        test_data, indices = self.get_samples(xda=xda, **self.config["testing"])
        if "sample_indices" not in self.config["testing"]:
            self.overwrite_config({"testing": {"sample_indices": indices}})
        self.localtime.stop()

        # setup ESN from zarr
        self.localtime.start("Read ESN Zarr Store")
        esn = from_zarr(**self.config["esn_weights"])
        self.localtime.stop()

        # make predictions
        self.localtime.start("Make Test Predictions")
        for i, tester in enumerate(test_data):
            xds = esn.test(
                    tester,
                    n_steps=self.config["testing"]["n_steps"],
                    n_spinup=self.config["testing"]["n_spinup"]
                    )
            xds.to_zarr(join(self.output_directory, f"test-{i}.zarr"), mode="w")

        self.localtime.stop()

        self.walltime.stop()


    def get_samples(self, xda, n_samples, n_steps, n_spinup, random_seed=None, sample_indices=None):
        """Pull random samples from macro_training or test dataset

        Args:
            xda (xarray.DataArray): with the full chunk of data to pull samples from
            n_samples (int): number of samples to grab
            n_steps (int): number of steps to make in sample prediction
            n_spinup (int): number of spinup steps before prediction
            random_seed (int, optional): RNG seed for grabbing temporal indices of random samples
            samples_indices (list, optional): the temporal indices denoting the start of the prediction period (including spinup), if provided then do not get a random sample of indices first

        Returns:
            samples (list of xarray.DataArray): with each separate sample trajectory
            sample_indices (list of int): with the initial conditions for the start of prediction phase, not the start of spinup
        """

        if sample_indices is None:
            sample_indices = self.get_sample_indices(
                    len(xda["time"]),
                    n_samples,
                    n_steps,
                    n_spinup,
                    random_seed)

        else:
            assert len(sample_indices) == n_samples, f"Driver.get_samples: found different values for len(sample_indices) and n_samples"

        samples = [xda.isel(time=slice(ridx-n_spinup, ridx+n_steps+1))
                   for ridx in sample_indices]

        return samples, sample_indices


    def get_sample_indices(self, data_length, n_samples, n_steps, n_spinup, random_seed):
        """Get random sample indices from dataset (without replacement) denoting initial conditions for training, validation, or testing

        Args:
            data_length (int): length of the dataseries along the time dimension
            n_samples (int): number of samples to grab
            n_steps (int): number of steps to make in sample prediction
            n_spinup (int): number of spinup steps before prediction
            random_seed (int, optional): RNG seed for grabbing temporal indices of random samples

        Returns:
            sample_indices (list): with integer indices denoting prediction initial conditions, not start of spinup
        """
        rstate = np.random.RandomState(seed=random_seed)
        n_valid = data_length - (n_steps + n_spinup)
        sample_indices = rstate.choice(n_valid, n_samples, replace=False)

        # add spinup here to get initial condition of the prediction, not including spinup
        sample_indices = list(int(x+n_spinup) for x in sample_indices)
        return sample_indices


    def _make_output_directory(self, out_dir):
        """Make provided output directory. If none given, make a unique directory:
        output-{self.name}-XX, where XX is 00->99

        Args:
            out_dir (str or None): path to dump output, or None for default

        Sets Attributes:
            out_dir (str): path to created output directory
        """
        if out_dir is None:

            # make a unique default directory
            i=0
            out_dir = f"output-{self.name}-{i:02d}"
            while os.path.isdir(out_dir):
                if i>99:
                    raise ValueError("Hit max number of default output directories...")
                out_dir = f"output-{self.name}-{i:02d}"
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
        self.logname = f'{self.name}_logger'
        self.logger = logging.getLogger(self.logname)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.logfile)
        fmt = logging.Formatter(style='{')
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)


    def set_config(self, config):
        """Read the nested parameter dictionary or take it directly, and write a copy for
        reference in the output_directory.

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
        """Overwrite specific parameters with the values in the nested dict new_config, e.g.

        new_config = {'esn':{'n_reservoir':1000}}

        will overwrite driver.config['esn']['n_reservoir'] with 1000, without having
        to recreate the big gigantic dictionary again.

        Args:
            new_config (dict): nested dictionary with values to overwrite object's parameters with

        Sets Attribute:
            config (dict): with the nested dictionary based on the input config file
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
                "testing": self.get_samples,
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
