import pytest

import os
from os.path import join
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import dask.array as darray

from smt.utils.design_space import DesignSpace

from xesn.cost import CostFunction
from xesn.xdata import XData
from xesn.driver import Driver
from xesn.test.xdata import test_data
from xesn.test.driver import lazy_driver

@pytest.fixture(scope="function")
def lazy_macro_driver(lazy_driver, test_data):
    driver, test_data = lazy_driver
    data = XData(**driver.config["xdata"])
    xda = data.setup(mode="macro_training")
    macro_data, indices = driver.get_samples(xda=xda, **driver.config["macro_training"]["forecast"])

    train_data = data.setup(mode="training")
    yield driver, train_data, macro_data
    # Do we need to rmtree(driver.output_directory)


def test_init(lazy_macro_driver):
    driver, train_data, macro_data = lazy_macro_driver

    cf = CostFunction(driver.ESN, train_data, macro_data, driver.config)

    assert cf.ESN == driver.ESN
    assert_array_equal(cf.train_data, train_data)
    for mdt, mde in zip(cf.macro_data, macro_data):
        assert_array_equal(mdt, mde)
    assert cf.config == driver.config


@pytest.mark.parametrize(
        # For some reason... n_parallel=1 makes design.sample_valid hang
        "n_parallel", [4]
    )
def test_eval(lazy_macro_driver, n_parallel):
    driver, train_data, macro_data = lazy_macro_driver

    cf = CostFunction(driver.ESN, train_data, macro_data, driver.config)

    bounds = np.array(list(driver.config["macro_training"]["parameters"].values()))
    design = DesignSpace(bounds)
    assert_array_equal(bounds, design.get_num_bounds())

    x_macro, _ = design.sample_valid_x(n_parallel)
    assert x_macro.shape == (n_parallel, len(bounds))
    cf(x_macro)
