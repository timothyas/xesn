
import os
from os.path import join
import yaml
import logging
import inspect
from contextlib import redirect_stdout

import numpy as np

from .esn import ESN
from .io import from_zarr
from .lazyesn import LazyESN
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
            - "xdata" with options passed to :meth:`XData.__init__`
            - "esn" or "lazyesn" with options passed to :meth:`ESN.__init__` or :meth:`LazyESN.__init__`
            - "training" with options passed to :meth:`ESN.train` or :meth:`LazyESN.train`
        """

        self.walltime.start("Starting Micro Calibration")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        xda = data.setup(mode="training")
        self.localtime.stop()

        # setup ESN
        self.localtime.start(f"Building {self.esn_name}")
        esn = self.ESN(**self.config[self.esn_name])
        esn.build()
        self.localtime.stop()

        self.localtime.start(f"Training {self.esn_name}")
        esn.train(xda, **self.config["training"])
        self.localtime.stop()

        self.localtime.start(f"Storing {self.esn_name} Weights")
        ds = esn.to_xds()
        ds.to_zarr(join(self.output_directory, f"{self.esn_name}-weights.zarr"), mode="w")
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


    def run_macro_calibration(self):
        """Perform Bayesian optimization on macro-scale ESN parameters using surrogate modeling toolbox.
        """

        self.walltime.start("Starting Macro Calibration")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        xda = data.setup(mode="validation")
        macro_data = self.get_samples("validation", xda=xda, **self.config["validation"])
        xda = data.setup(mode="training")
        self.localtime.stop()

        # create cost function
        self.localtime.start("Setting up cost function")
        cf = CostFunction(self.ESN, xda, macro_data, self.config)
        self.localtime.stop()

        # optimize
        self.localtime.start("Starting Bayesian Optimization")
        p_opt = optimize(self.config["optim"]["parameters"],
                         self.config["optim"]["transformations"],
                         cost_function,
                         **self.config["ego"])
        self.localtime.stop()

        self.walltime.stop()


    def run_test(self):
        """Perform ESN training, learn the readout matrix weights.

        Required Parameter Sections:
            - "xdata" with options passed to :meth:`XData.__init__`
            - "esn_weights" with options passed to :func:`from_zarr`
            - "testing" with options passed to :meth:`get_samples`, except "mode" and "xda"
        """

        self.walltime.start("Starting Testing")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.config["xdata"])
        xda = data.setup(mode="testing")
        self.localtime.stop()

        # pull samples from data
        self.localtime.start("Get Test Samples")
        test_data = self.get_samples("testing", xda=xda, **self.config["testing"])
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


    def get_samples(self, mode, xda, n_samples, n_steps, n_spinup, random_seed=None, sample_indices=None):
        """Pull random samples from validation or test dataset

        Args:
            mode (str): indicating validation or test
            xda (xarray.DataArray): with the full chunk of data to pull samples from
            n_samples (int): number of samples to grab
            n_steps (int): number of steps to make in validation/test prediction
            n_spinup (int): number of spinup steps before prediction
            random_seed (int, optional): RNG seed for grabbing temporal indices of random samples
            samples_indices (list, optional): the temporal indices denoting the start of the prediction period (including spinup)

        Returns:
            testers (list of xarray.DataArray): with each separate validation/test sample
        """

        self._set_sample_indices(
                mode,
                len(xda.time),
                n_samples,
                n_steps,
                n_spinup,
                random_seed,
                sample_indices)

        testers = [xda.isel(time=slice(ridx, ridx+n_steps+n_spinup+1))
                   for ridx in self.config[mode]["sample_indices"]]
        return testers


    def _set_sample_indices(self, mode, data_length, n_samples, n_steps, n_spinup, random_seed, sample_indices):
        """If sample indices aren't provided, get them.

        Sets Attributes:
            sample_indices (list): with temporal indices denoting the start of the prediction period (including spinup)
        """

        if sample_indices is None:

            rstate = np.random.RandomState(seed=random_seed)
            n_valid = data_length - (n_steps + n_spinup)
            sample_indices = rstate.choice(n_valid, n_samples, replace=False)

        # make sure types are good to go
        sample_indices = list(int(x) for x in sample_indices)
        self.overwrite_config({mode: {"sample_indices": sample_indices}})


#    def run_macro_calibration(self):
#
#        self.walltime.start("Starting Macro Calibration")
#
#        # setup the data
#        data = XData(**self.config["xdata"])
#        xda = data.setup()
#
#        # define the loss function
#
#        # optimize
#
#        # Retrain (for now... need to dig into this)
#
#        self.walltime.stop("Total Walltime")


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
            with open(config, "r") as f:
                config = yaml.safe_load(f)

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
                self.print_log(f"Driver.overwrite_config: Overwriting driver.config['{s}']['{key}'] with {val}")
                config[s][key] = val

        # Overwrite our copy of config.yaml in output_dir and reset attr
        self.set_config(config)


    def print_log(self, *args, **kwargs):
        """Print to log file as specified in :attr:`logname`"""
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                print(*args, **kwargs)


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
                "validation": None,
                "testing": self.get_samples,
                "compute": None,
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
