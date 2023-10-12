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
from xesn.test.xdata import test_data
from xesn.test.esn import test_data as eager_data

@pytest.fixture(scope="function")
def config_dict():
    c = {
            "xdata": {
                "zstore_path": "blah.zarr",
                "field_name": "blah",
            },
            "lazyesn": {
                "n_reservoir": 500,
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
                },
            }

        driver.overwrite_config(expected)

        # make sure these changed
        assert driver.config["xdata"]["zstore_path"] == expected["xdata"]["zstore_path"]
        assert driver.config["lazyesn"]["n_reservoir"] == expected["lazyesn"]["n_reservoir"]

        # but make sure this one didn't
        assert driver.config["xdata"]["field_name"] == config_dict["xdata"]["field_name"]

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


@pytest.fixture(scope="function")
def eager_driver(test_data):
    driver = Driver(join(os.path.dirname(__file__), "config-eager.yaml"),
                    output_directory="test-eager-driver")
    fname = "eager-xdata.zarr"
    edata = test_data.isel(y=0, z=0).chunk({"x":-1}).to_dataset().to_zarr(fname)
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
        driver.run_micro_calibration()
        assert len(glob(f"{driver.output_directory}/*esn-weights.zarr")) == 1


    @pytest.mark.parametrize(
            "this_driver", ("eager_driver", "lazy_driver")
        )
    def test_testing(self, this_driver, request):

        driver, _ = request.getfixturevalue(this_driver)
        driver.run_micro_calibration()

        driver.run_test()
