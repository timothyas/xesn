import pytest

import os
from os.path import join
import yaml
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

import dask.array as darray

from esnpy.driver import Driver
from xdata import test_data

@pytest.fixture(scope="module")
def config_dict():
    c = {
            "xdata": {
                "zstore_path": "blah.zarr",
                "field_name": "blah",
            },
            "LazyESN": {
                "n_reservoir": 500,
            },
        }
    yield c

@pytest.fixture(scope="module")
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

class TestDriver():

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

    def test_output_dataset(self, config_dict):
        with pytest.raises(TypeError):
            driver = Driver(config_dict, output_dataset_filename=None)

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

    def test_overwrite_params(self, test_driver, config_dict):

        driver = test_driver
        assert driver.params["xdata"]["zstore_path"] == config_dict["xdata"]["zstore_path"]
        assert driver.params["LazyESN"]["n_reservoir"] == config_dict["LazyESN"]["n_reservoir"]

        expected = {
                "xdata": {
                    "zstore_path": "new.zarr",
                },
                "LazyESN": {
                    "n_reservoir": 100,
                },
            }

        driver.overwrite_params(expected)

        # make sure these changed
        assert driver.params["xdata"]["zstore_path"] == expected["xdata"]["zstore_path"]
        assert driver.params["LazyESN"]["n_reservoir"] == expected["LazyESN"]["n_reservoir"]

        # but make sure this one didn't
        assert driver.params["xdata"]["field_name"] == config_dict["xdata"]["field_name"]

    def test_params_type(self, test_driver):
        driver = test_driver
        with pytest.raises(TypeError):
            driver.set_params(["blah", "blah", "blah"])


    def test_bad_section(self, test_driver, config_dict):
        c = config_dict.copy()
        c["blah"] = {"a": 1}
        driver = test_driver
        with pytest.raises(KeyError):
            driver.set_params(c)


    def test_bad_option(self, test_driver, config_dict):
        c = config_dict.copy()
        c["xdata"]["blah"] = None
        driver = test_driver
        with pytest.raises(KeyError):
            driver.set_params(c)


@pytest.fixture(scope="function")
def train_driver(test_data):

    driver = Driver("config-train.yaml")
    yield driver, test_data
    rmtree(driver.output_directory)


class TestDriverTraining():
    def test_micro_training(self, train_driver):

        driver, test_data = train_driver
        driver.run_micro_calibration()
        assert os.path.basename(driver.output_dataset_filename) in os.listdir(driver.output_directory)
