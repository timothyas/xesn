
import os
from os.path import join
import yaml
import logging
import inspect
from contextlib import redirect_stdout

#TODO remove these if test is added to esn methods
import numpy as np
import xarray as xr

from .esn import ESN
from .io import from_zarr
from .lazyesn import LazyESN
from .timer import Timer
from .xdata import XData

class Driver():
    name                    = "driver"
    output_directory        = None
    walltime                = None
    localtime               = None

    def __init__(self,
                 config,
                 output_directory=None):

        self.make_output_directory(output_directory)
        self.create_logger()
        self.set_params(config)

        # Look for ESN or LazyESN
        if "esn" in self.params.keys():
            self.ESN = ESN
        elif "lazyesn" in self.params.keys():
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

        self.walltime.start("Starting Micro Calibration")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.params["xdata"])
        xda = data.setup(mode="training")
        self.localtime.stop()

        # setup ESN
        self.localtime.start(f"Building {self.esn_name}")
        esn = self.ESN(**self.params[self.esn_name])
        esn.build()
        self.localtime.stop()

        self.localtime.start(f"Training {self.esn_name}")
        array = xda.data if "lazy" in self.esn_name else xda.values
        esn.train(array, **self.params["training"])
        self.localtime.stop()

        self.localtime.start(f"Storing {self.esn_name} Weights")
        ds = esn.to_xds()
        ds.to_zarr(join(self.output_directory, f"{self.esn_name}-weights.zarr"))
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


    def run_test(self):

        self.walltime.start("Starting Testing")

        # setup the data
        self.localtime.start("Setting up Data")
        data = XData(**self.params["xdata"])
        xda = data.setup(mode="testing")
        self.localtime.stop()

        # pull samples from data
        self.localtime.start("Get Test Samples")
        test_data = self.get_samples("testing", test_data=xda, **self.params["testing"])
        self.localtime.stop()

        # setup ESN from zarr
        self.localtime.start("Read ESN Zarr Store")
        esn = from_zarr(**self.params["esn_weights"])
        self.localtime.stop()

        # make predictions
        self.localtime.start("Make Test Predictions")
        coords = {key: test_data[0][key] for key in test_data[0].dims if key != "time"}
        coords["time"] = test_data[0].time.isel(time=slice(self.params["testing"]["n_spinup"], None))
        dims = test_data[0].dims
        for i, tester in enumerate(test_data):
            xds = xr.Dataset()
            array = tester.data if "lazy" in self.esn_name else tester.values
            prediction = esn.predict(
                    array,
                    n_steps=self.params["testing"]["n_steps"],
                    n_spinup=self.params["testing"]["n_spinup"]
                    )
            xds["truth"] = xr.DataArray(tester.isel(time=slice(self.params["testing"]["n_spinup"], None)), coords=coords, dims=dims)
            xds["prediction"] = xr.DataArray(prediction, coords=xds.truth.coords, dims=xds.truth.dims)
            xds.to_zarr(join(self.output_directory, f"test-{i}.zarr"))

        self.localtime.stop()

        self.walltime.stop()


    def get_samples(self, mode, test_data, n_samples, n_steps, n_spinup, random_seed=None, sample_indices=None):

        self.set_sample_indices(
                mode,
                len(test_data.time),
                n_samples,
                n_steps,
                n_spinup,
                random_seed,
                sample_indices)

        testers = [test_data.isel(time=slice(ridx, ridx+n_steps+n_spinup+1))
                   for ridx in self.params[mode]["sample_indices"]]
        return testers


    def set_sample_indices(self, mode, data_length, n_samples, n_steps, n_spinup, random_seed, sample_indices):

        if sample_indices is None:

            rstate = np.random.RandomState(seed=random_seed)
            n_valid = data_length - (n_steps + n_spinup)
            sample_indices = rstate.choice(n_valid, n_samples, replace=False)

        # make sure types are good to go
        sample_indices = list(int(x) for x in sample_indices)
        self.overwrite_params({mode: {"sample_indices": sample_indices}})


#    def run_macro_calibration(self):
#
#        self.walltime.start("Starting Macro Calibration")
#
#        # setup the data
#        data = XData(**self.params["xdata"])
#        xda = data.setup()
#
#        # define the loss function
#
#        # optimize
#
#        # Retrain (for now... need to dig into this)
#
#        self.walltime.stop("Total Walltime")


    def make_output_directory(self, out_dir):
        """Make provided output directory. If none given, make a unique directory:
            output-{self.name}-XX
        XX is 00->99

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


    def create_logger(self):
        """Create a logfile and logger for writing all output to

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


    def set_params(self, config):
        """Read the nested parameter dictionary or take it directly, and write a copy for
        reference in the output_directory.

        Args:
            config (str or dict): filename (path) to the configuration yaml file, or nested dictionary with parameters

        Sets Attribute:
            params (dict): with a big nested dictionary with all parameters
        """

        if isinstance(config, str):
            with open(config, "r") as f:
                params = yaml.safe_load(f)

        elif isinstance(config, dict):
            params = config

        else:
            raise TypeError(f"Driver.set_params: Unrecognized type for experiment config, must be either yaml filename (str) or a dictionary with parameter values")

        # make the section names lower case
        lparams = {}
        for key in params.keys():
            lparams[key.lower()] = params[key]

        self._check_config_options(lparams)
        self.params = lparams

        outname = os.path.join(self.output_directory, "config.yaml")
        with open(outname, "w") as f:
            yaml.dump(self.params, stream=f)


    def overwrite_params(self, new_params):
        """Overwrite specific parameters with the values in the nested dict new_params, e.g.

        new_params = {'model':{'n_reservoir':1000}}

        will overwrite driver.params['model']['n_reservoir'] with 1000, without having
        to recreate the big gigantic dictionary again.

        Args:
            new_params (dict): nested dictionary with values to overwrite object's parameters with

        Sets Attribute:
            params (dict): with the nested dictionary based on the input config file
        """

        params = self.params.copy()
        for section, this_dict in new_params.items():
            for key, val in this_dict.items():
                s = section.lower()
                self.print_log(f"Driver.overwrite_params: Overwriting driver.params['{s}']['{key}'] with {val}")
                params[s][key] = val

        # Overwrite our copy of config.yaml in output_dir and reset attr
        self.set_params(params)


    def print_log(self, *args, **kwargs):
        """Print to log file"""
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                print(*args, **kwargs)


    def _check_config_options(self, params):
        """Make sure we recognize each configuration section name, and each option name.
        No type or value checking

        Args:
            params (dict): the big nested options dictionary
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
        for key in params.keys():
            try:
                assert key in expected.keys()
            except:
                bad_sections.append(key)

        if len(bad_sections)>0:
            raise KeyError(f"Driver._check_config_options: unrecognized config section(s): {bad_sections}")

        # Check options in each section
        for section in params.keys():
            Func = expected[section]
            if Func is not None:
                kw, *_ = inspect.getfullargspec(Func)
                for key in params[section].keys():
                    try:
                        assert key in kw
                    except:
                        raise KeyError(f"Driver._check_config_options: unrecognized option {key} in section {section}")
