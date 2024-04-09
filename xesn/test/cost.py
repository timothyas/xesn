import pytest

import numpy as np
from numpy.testing import assert_array_equal

from smt.utils.design_space import DesignSpace

from xesn.cost import CostFunction
from xesn.xdata import XData
from xesn.utils import get_samples
from xesn.test.xdata import test_data
from xesn.test.driver import eager_driver, lazy_driver
from xesn import _use_cupy

pytestmark = pytest.mark.skipif(_use_cupy, reason="writing directories with driver causes unexpected failures")

@pytest.fixture(scope="function")
def eager_macro_driver(eager_driver):
    driver, eager_data = eager_driver
    data = XData(**driver.config["xdata"])
    xda = data.setup(mode="macro_training")
    macro_data, indices = get_samples(xda=xda, **driver.config["macro_training"]["forecast"])
    train_data = data.setup(mode="training")
    yield driver, train_data, macro_data


@pytest.fixture(scope="function")
def lazy_macro_driver(lazy_driver, test_data):
    driver, test_data = lazy_driver
    data = XData(**driver.config["xdata"])
    xda = data.setup(mode="macro_training")
    macro_data, indices = get_samples(xda=xda, **driver.config["macro_training"]["forecast"])

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


@pytest.mark.parametrize(
        "use_test_data", (True, False),
    )
@pytest.mark.parametrize(
        "this_driver", ("eager_macro_driver", "lazy_macro_driver"),
    )
def test_evaluate(use_test_data, this_driver, request):
    driver, train_data, macro_data = request.getfixturevalue(this_driver)

    cf = CostFunction(driver.ESN, train_data, macro_data, driver.config, test_data=macro_data)

    parameters = {
        "input_factor": .5,
        "adjacency_factor": .5,
        "bias_factor": .5,
        "tikhonov_parameter": 1e-6,
    }

    xds = cf.evaluate(parameters, use_test_data=use_test_data)

    for key in [
        "truth", "prediction", "nrmse",
        "psd_truth", "psd_prediction", "psd_nrmse"]:
        assert key in xds
        assert "ftime" in xds[key].dims
        assert "sample" in xds[key].dims

    assert len(xds["sample"]) == driver.config["macro_training"]["forecast"]["n_samples"]
