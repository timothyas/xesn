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
from xesn.test.driver import eager_driver, lazy_driver

@pytest.fixture(scope="function")
def eager_macro_driver(eager_driver):
    driver, eager_data = eager_driver
    data = XData(**driver.config["xdata"])
    xda = data.setup(mode="macro_training")
    macro_data, indices = driver.get_samples(xda=xda, **driver.config["macro_training"]["forecast"])
    train_data = data.setup(mode="training")
    yield driver, train_data, macro_data


@pytest.fixture(scope="function")
def lazy_macro_driver(lazy_driver, test_data):
    driver, test_data = lazy_driver
    data = XData(**driver.config["xdata"])
    xda = data.setup(mode="macro_training")
    macro_data, indices = driver.get_samples(xda=xda, **driver.config["macro_training"]["forecast"])

    train_data = data.setup(mode="training")
    yield driver, train_data, macro_data


@pytest.mark.parametrize(
        "this_driver", ("eager_macro_driver", "lazy_macro_driver"),
    )
def test_init(this_driver, request):
    driver, train_data, macro_data = request.getfixturevalue(this_driver)

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
@pytest.mark.parametrize(
        "this_driver", ("eager_macro_driver", "lazy_macro_driver"),
    )
def test_eval(n_parallel, this_driver, request):
    driver, train_data, macro_data = request.getfixturevalue(this_driver)

    cf = CostFunction(driver.ESN, train_data, macro_data, driver.config)

    bounds = np.array(list(driver.config["macro_training"]["parameters"].values()))
    design = DesignSpace(bounds)
    assert_array_equal(bounds, design.get_num_bounds())

    x_macro, _ = design.sample_valid_x(n_parallel)
    assert x_macro.shape == (n_parallel, len(bounds))
    cf(x_macro)
