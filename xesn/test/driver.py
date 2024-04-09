import pytest

import os
from os.path import join
from glob import glob
import yaml
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

import dask.array as darray

from xesn.driver import Driver
from xesn.utils import get_samples, get_sample_indices
from xesn.test.xdata import test_data
from xesn import _use_cupy

pytestmark = pytest.mark.skipif(_use_cupy, reason="writing directories with driver causes unexpected failures")


@pytest.fixture(scope="function")
def config_dict():
    c = {
            "xdata": {
                "zstore_path": "blah.zarr",
                "field_name": "blah",
            },
            "lazyesn": {
                "n_reservoir": 500,
                "input_kwargs": {
                    "factor": 1.0,
                    "distribution": "uniform",
                    "normalization": "multiply",
                    "is_sparse": False,
                },
            },
            "testing": {
                "n_samples": 5,
                "n_steps": 2,
                "n_spinup": 3,
                "random_seed": 0,
            },
        }
    yield c

@pytest.fixture(scope="function")
def config_yaml(config_dict):
    c = config_dict

    cname = "config.yaml"
    with open(cname, "w") as f:
        yaml.dump(c, stream=f)
    yield cname
    os.remove(cname)

@pytest.fixture(scope="function")
def test_driver(config_dict):
    driver = Driver(config_dict)
    yield driver
    rmtree(driver.output_directory)

@pytest.fixture(scope="function")
def test_yaml_driver(config_yaml):
    driver = Driver(config_yaml)
    yield driver
    rmtree(driver.output_directory)

class TestDriverBasic():

    @pytest.mark.parametrize(
            "tester", ("test_driver", "test_yaml_driver"),
        )
    def test_basic(self, tester, request):
        driver = request.getfixturevalue(tester)

        for key in ["output_directory", "walltime", "localtime"]:
            assert getattr(driver, key) is not None
        assert driver.logfile == join(driver.output_directory, "stdout.log")
        assert not driver.walltime.is_running()
        assert not driver.localtime.is_running()

        print(str(driver))
        print(driver.__repr__)

    def test_default_output_directory(self, config_dict):

        for i in range(100):
            driver = Driver(config_dict)
            assert driver.output_directory == f"output-driver-{i:02d}"
            assert driver.output_directory in os.listdir()

        with pytest.raises(ValueError):
            Driver(config_dict)

        for i in range(100):
            rmtree(f"output-driver-{i:02d}")

    def test_custom_output_directory(self, config_dict):

        expected = "test-dir"
        driver = Driver(config_dict, output_directory=expected)
        assert driver.output_directory == expected
        assert expected in os.listdir()

        rmtree(expected)

    def test_overwrite_config(self, test_driver, config_dict):

        driver = test_driver
        assert driver.config["xdata"]["zstore_path"] == config_dict["xdata"]["zstore_path"]
        assert driver.config["lazyesn"]["n_reservoir"] == config_dict["lazyesn"]["n_reservoir"]

        expected = {
                "xdata": {
                    "zstore_path": "new.zarr",
                },
                "lazyesn": {
                    "n_reservoir": 100,
                    "input_kwargs": {
                        "factor": 0.5,
                        "random_seed": 0,
                    },
                    "adjacency_kwargs": {
                        "factor": 1.0,
                        "distribution": "gaussian",
                        "normalization": "svd",
                        "is_sparse": True,
                        "density": 0.1,
                    },
                },
                "esn_weights": {
                    "zstore_path": "path/to/weights.zarr",
                },
                "macro_training": {
                    "parameters": {
                        "input_factor": [1e-4, 1.0],
                        "tikhonov_parameter": [1e-9, 1.0],
                    },
                },
            }

        driver.overwrite_config(expected)

        # make sure these changed
        assert driver.config["xdata"]["zstore_path"] == expected["xdata"]["zstore_path"]
        assert driver.config["lazyesn"]["n_reservoir"] == expected["lazyesn"]["n_reservoir"]
        assert driver.config["lazyesn"]["input_kwargs"]["factor"] == expected["lazyesn"]["input_kwargs"]["factor"]
        assert driver.config["lazyesn"]["input_kwargs"]["random_seed"] == expected["lazyesn"]["input_kwargs"]["random_seed"]
        assert driver.config["esn_weights"]["zstore_path"] == expected["esn_weights"]["zstore_path"]
        assert driver.config["macro_training"] == expected["macro_training"]
        assert driver.config["lazyesn"]["adjacency_kwargs"] == expected["lazyesn"]["adjacency_kwargs"]

        # but make sure this one didn't
        assert driver.config["xdata"]["field_name"] == config_dict["xdata"]["field_name"]
        assert driver.config["lazyesn"]["input_kwargs"]["distribution"] == config_dict["lazyesn"]["input_kwargs"]["distribution"]

    def test_config_type(self, test_driver):
        driver = test_driver
        with pytest.raises(TypeError):
            driver.set_config(["blah", "blah", "blah"])


    def test_bad_section(self, test_driver, config_dict):
        c = config_dict.copy()
        c["blah"] = {"a": 1}
        driver = test_driver
        with pytest.raises(KeyError):
            driver.set_config(c)


    def test_bad_option(self, test_driver, config_dict):
        c = config_dict.copy()
        c["xdata"]["blah"] = None
        driver = test_driver
        with pytest.raises(KeyError):
            driver.set_config(c)

    def test_case(self, test_driver):
        driver = test_driver
        c = driver.config.copy()
        expected = driver.config.copy()
        c["XDATA"] = c["xdata"]
        del c["xdata"]

        driver.set_config(c)
        assert driver.config == expected

        driver.overwrite_config(c)
        assert driver.config == expected

    def test_samples(self, test_driver, test_data):
        driver = test_driver
        indices = get_sample_indices(len(test_data["time"]), **driver.config["testing"])

        # run it once
        testers, new_indices = get_samples(test_data, **driver.config["testing"])
        assert_array_equal(indices, new_indices)
        assert len(testers) == driver.config["testing"]["n_samples"]

        # make sure it's the same when we give the indices
        testers2, _ = get_samples(
                test_data,
                driver.config["testing"]["n_samples"],
                driver.config["testing"]["n_steps"],
                driver.config["testing"]["n_spinup"],
                sample_indices=indices)
        for t1,t2 in zip(testers, testers2):
            assert_array_equal(t1,t2)

        # now raise a problem when different n_samples and len(indices)
        with pytest.raises(AssertionError):
            get_samples(
                    test_data,
                    driver.config["testing"]["n_samples"],
                    driver.config["testing"]["n_steps"],
                    driver.config["testing"]["n_spinup"],
                    sample_indices=indices[:2])


    def test_load(self, test_driver):
        """Test this addition to yaml.load to recognize floats more intuitively"""

        driver = test_driver
        c = driver.load(join(os.path.dirname(__file__), "config-lazy.yaml"))
        assert tuple(c["xdata"]["dimensions"]) == ("x", "y", "z", "time")
        assert tuple(c["xdata"]["subsampling"]["time"]["training"]) == (None, 100, None)
        assert isinstance(c["xdata"]["normalization"]["bias"], float)
        assert np.abs(c["xdata"]["normalization"]["bias"]) < 1e-15
        assert isinstance(c["lazyesn"]["n_reservoir"], int)
        assert isinstance(c["lazyesn"]["tikhonov_parameter"], float)
        assert_allclose(c["lazyesn"]["tikhonov_parameter"], 1.e-6)
        assert isinstance(c["lazyesn"]["persist"], bool)
        assert c["lazyesn"]["persist"]
        assert c["training"]["batch_size"] is None
        assert isinstance(c["macro_training"]["cost_upper_bound"], float)
        assert_allclose(c["macro_training"]["cost_upper_bound"], 1e9)
        assert isinstance(c["macro_training"]["parameters"]["input_factor"][0], float)
        assert isinstance(c["macro_training"]["parameters"]["input_factor"][1], float)
        assert_allclose(c["macro_training"]["parameters"]["input_factor"], [0.01, 100])



@pytest.fixture(scope="function")
def eager_driver(test_data):
    driver = Driver(join(os.path.dirname(__file__), "config-eager.yaml"),
                    output_directory="test-eager-driver")
    fname = "eager-xdata.zarr"
    edata = test_data.isel(y=0, z=0).drop_vars(["y", "z"]).chunk({"x":-1}).to_dataset().to_zarr(fname)
    yield driver, edata
    rmtree(driver.output_directory)
    rmtree(fname)


@pytest.fixture(scope="function")
def lazy_driver(test_data):

    driver = Driver(join(os.path.dirname(__file__), "config-lazy.yaml"),
                    output_directory="test-lazy-driver")
    yield driver, test_data
    rmtree(driver.output_directory)


class TestDriverCompute():

    @pytest.mark.parametrize(
            "this_driver", ("eager_driver", "lazy_driver")
        )
    def test_micro_training(self, this_driver, request):

        driver, _ = request.getfixturevalue(this_driver)
        driver.run_training()
        assert len(glob(f"{driver.output_directory}/*esn-weights.zarr")) == 1


    @pytest.mark.parametrize(
            "this_driver", ("eager_driver", "lazy_driver")
        )
    def test_testing(self, this_driver, request):

        driver, _ = request.getfixturevalue(this_driver)
        driver.run_training()

        driver.run_test()

        # make sure sample indices got written out
        new_config = f"{driver.output_directory}/config.yaml"
        with open(new_config, "r") as f:
            nc = yaml.safe_load(f)

        assert all(x == y for x,y in zip(
            driver.config["testing"]["sample_indices"],
            nc["testing"]["sample_indices"]))


    @pytest.mark.parametrize(
            "this_driver", ("eager_driver", "lazy_driver")
        )
    def test_macro_training(self, this_driver, request):
        if _use_cupy:
            pytest.skip("Macro training unsupported on GPUs")

        driver, _ = request.getfixturevalue(this_driver)
        driver.run_macro_training()

        # make sure sample indices got written out
        new_config = f"{driver.output_directory}/config.yaml"
        with open(new_config, "r") as f:
            nc = yaml.safe_load(f)

        assert all(x == y for x,y in zip(
            driver.config["macro_training"]["forecast"]["sample_indices"],
            nc["macro_training"]["forecast"]["sample_indices"]))

        # make sure optim file is there
        config_optim = f"{driver.output_directory}/config-optim.yaml"
        assert os.path.isfile(config_optim)
