import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from xesn.optim import transform, inverse_transform

@pytest.fixture(scope="function")
def transform_params():

    params = {
            "input_factor"      : 0.5,
            "adjacency_factor"  : 0.5,
            "bias_factor"       : 0.5}

    transforms ={
            "input_factor"      : "log10",
            "adjacency_factor"  : "log"}

    yield params, transforms

@pytest.fixture(scope="function")
def transform_bounds(transform_params):
    params, transforms = transform_params
    bounds = {}
    for key in params.keys():
        bounds[key] = (1.e-4, 2.)

    yield bounds, transforms


@pytest.mark.parametrize(
        "transform_inputs", ("transform_params", "transform_bounds"),
    )
def test_transform(transform_inputs, request):

    params, transforms = request.getfixturevalue(transform_inputs)
    ptest = transform(params, transforms)

    assert_allclose(np.array(ptest["input_factor"]), np.log10(np.array(params["input_factor"])))
    assert_allclose(np.array(ptest["adjacency_factor"]), np.log(np.array(params["adjacency_factor"])))
    assert_allclose(np.array(ptest["bias_factor"]), np.array(params["bias_factor"]))


@pytest.mark.parametrize(
        "transform_inputs", ("transform_params", "transform_bounds"),
    )
def test_inverse_transform(transform_inputs, request):

    params, transforms = request.getfixturevalue(transform_inputs)
    ptest = inverse_transform(params, transforms)

    assert_allclose(np.array(ptest["input_factor"]), 10.** np.array(params["input_factor"]))
    assert_allclose(np.array(ptest["adjacency_factor"]), np.exp(np.array(params["adjacency_factor"])))
    assert_allclose(np.array(ptest["bias_factor"]), np.array(params["bias_factor"]))

@pytest.mark.parametrize(
        "transformer", (transform, inverse_transform)
    )
def test_no_key(transform_params, transformer):

    params, transforms = transform_params
    t = transforms.copy()
    t["blah"] = "log"
    with pytest.raises(KeyError):
        transformer(params, t)


@pytest.mark.parametrize(
        "transformer", (transform, inverse_transform)
    )
def test_notimplemented(transform_params, transformer):

    params, transforms = transform_params
    t = transforms.copy()
    t["input_factor"] = "something_crazy"
    with pytest.raises(NotImplementedError):
        transformer(params, t)
