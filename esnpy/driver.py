
import os
import yaml
import logging
from contextlib import redirect_stdout

from .xdata import XData
from .lazyesn import LazyESN
from .timer import Timer

# TODO:
# - create client / dask stuff
# - how to choose between different models. is it enough to have the title of a section be different? have to look for recognized names
# - make consistent validation vs macro training and training vs micro training

class Driver():
    name                    = "driver"
    output_directory        = None
    output_datset_filename  = None
    walltime                = None
    localtime               = None

    def __init__(self,
                 config,
                 output_directory=None,
                 output_dataset_filename="results.zarr"):

        try:
            assert isinstance(output_dataset_filename, str)
        except:
            raise TypeError("Driver.__init__: output_dataset_filename must be a string, denoting zarr store path")

        self.make_output_directory(output_directory)
        self.create_logger()
        self.output_dataset_filename = os.path.join(self.output_directory, output_dataset_filename)
        self.set_params(config)

        self.walltime = Timer(filename=self.logfile)
        self.localtime = Timer(filename=self.logfile)

        self.print_log(" --- Driver Initialized --- \n")
        self.print_log(self)


    def __str__(self):
        mystr = "Driver\n"+\
                f"    {'output_directory:':<28s}{self.output_directory}\n"+\
                f"    {'logfile:':<28s}{self.logfile}\n"+\
                f"    {'output_dataset_filename:':<28s}{self.output_dataset_filename}"
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
        # TODO: how to choose between lazy or not
        self.localtime.start("Building ESN")
        esn = LazyESN(**self.params["LazyESN"])
        esn.build()
        self.localtime.stop()

        self.localtime.start("Training ESN")
        esn.train(xda.data, **self.params["training"])
        self.localtime.stop()

        self.localtime.start("Storing ESN Weights")
        ds = esn.to_xds()
        ds.to_zarr(self.output_dataset_filename)
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


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

        # TODO: determine if we really want a check like this,
        # or if it's enough to just have each section get passed to class initialization
        #self._check_config_options(params)

        outname = os.path.join(self.output_directory, "config.yaml")
        with open(outname, "w") as f:
            yaml.dump(params, stream=f)

        self.params = params


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
                self.print_log(f"Driver.overwrite_params: Overwriting driver.params['{section}']['{key}'] with {val}")
                params[section][key] = val

        # Overwrite our copy of config.yaml in output_dir and reset attr
        self.set_params(params)


    def print_log(self, *args, **kwargs):
        """Print to log file"""
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                print(*args, **kwargs)


# TODO: decide on this, probably going to delete it b/c it's extraneous code.
#    def _check_config_options(self, params):
#        """A really simple test, make sure we recognize each option name, that's it.
#
#        Args:
#            params (dict): the big nested options dictionary
#        """
#
#        this_dir = os.path.dirname(os.path.abspath(__file__))
#        fname = os.path.join(this_dir, "options.yaml")
#        with open(fname, "r") as f:
#            options = yaml.safe_load(f)
#
#        for section in params.keys():
#            for key in params[section].keys():
#                try:
#                    assert key in options[section]
#                except:
#                    raise KeyError(f"Driver.check_config_options: unrecognized parameter option {key} found in config section {section}")
