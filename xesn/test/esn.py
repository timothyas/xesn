import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

from xesn.esn import ESN
from xesn.io import from_zarr

class TestESN:
    n_input             = 3
    n_output            = 3
    n_reservoir         = 100
    n_train             = 500
    connectedness       = 5
    bias_factor         = 0.1
    leak_rate           = 0.5
    tikhonov_parameter  = 1e-6

    input_factor        = 0.1
    adjacency_factor    = 0.1

    @property
    def kw(self):
        return {key: getattr(self, key) for key in [
            "n_input", "n_output", "n_reservoir", "connectedness", "bias_factor", "leak_rate", "tikhonov_parameter", "input_factor", "adjacency_factor"]}

    equal_list  = ("n_input", "n_output", "n_reservoir")
    close_list  = ("input_factor", "adjacency_factor", "connectedness", "bias_factor", "leak_rate", "tikhonov_parameter")


class TestInit(TestESN):

    def test_basic(self):
        esn = ESN(**self.kw)
        str(esn)
        assert esn.__repr__() == str(esn)

        for key in self.kw.keys():

            expected = self.kw[key]
            test = getattr(esn, key)

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

    @pytest.mark.parametrize(
            "key, val, raises, error",
            [
                ("bias_factor", -1., pytest.raises, ValueError),
                ("connectedness", 101, pytest.raises, ValueError),
                ("connectedness", 90, pytest.warns, RuntimeWarning),
            ],
    )
    def test_errors(self, key, val, raises, error):
        kwargs = self.kw.copy()
        kwargs[key] = val
        with raises(error):
            esn = ESN(**kwargs)

@pytest.mark.parametrize(
        "distribution", ["uniform", "normal", "gaussian"],
)
@pytest.mark.parametrize(
        "factor, error", [(0.1, None), (0.5, ValueError)],
)
class TestVectors(TestESN):

    def test_bias_kwargs(self, distribution, factor, error):

        bkw = { "distribution"  : distribution,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if error is None:
            esn = ESN(**self.kw, bias_kwargs=bkw)
            for key, expected in bkw.items():
                assert esn.bias_kwargs[key] == expected

            esn.build()
            assert len(esn.bias_vector) == self.n_reservoir

        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, bias_kwargs=bkw)

@pytest.mark.parametrize(
        "distribution", ["uniform", "normal", "gaussian"],
)
@pytest.mark.parametrize(
        "factor, error", [(0.1, None), (0.5, ValueError)],
)
@pytest.mark.parametrize(
        "is_sparse", [True, False],
)
class TestMatrices(TestESN):

    @pytest.mark.parametrize(
            "normalization", ["multiply", "svd"],
    )
    def test_input_kwargs(self, distribution, normalization, is_sparse, factor, error):
        ikw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if is_sparse:
            ikw["density"] = 0.1

        if error is None:
            esn = ESN(**self.kw, input_kwargs=ikw)
            for key, expected in ikw.items():
                assert esn.input_kwargs[key] == expected

            esn.build()
            assert tuple(esn.Win.shape) == (self.n_reservoir, self.n_input)
        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, input_kwargs=ikw)


    @pytest.mark.parametrize(
            "normalization", ["multiply", "eig", "svd"],
    )
    def test_adjacency_kwargs(self, distribution, normalization, is_sparse, factor, error):
        akw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if is_sparse:
            akw["density"] = 0.1

        if error is None:
            esn = ESN(**self.kw, adjacency_kwargs=akw)
            for key, expected in akw.items():
                assert esn.adjacency_kwargs[key] == expected

            esn.build()
            assert tuple(esn.W.shape) == (self.n_reservoir, self.n_reservoir)
        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, adjacency_kwargs=akw)



@pytest.fixture(scope="module")
def test_data():
    rs = np.random.RandomState(0)
    tester = TestESN()

    datasets = {}

    time = np.arange(tester.n_train)
    for n_input in [3, 7]:
        datasets[n_input] = {
                "u": xr.DataArray(
                    rs.normal(size=(n_input, tester.n_train)),
                    coords={"x": np.arange(n_input), "time": time},
                    dims=("x", "time")),
                "y": xr.DataArray(
                    rs.normal(size=(tester.n_output, tester.n_train)),
                    coords={"x": np.arange(tester.n_output), "time": time},
                    dims=("x", "time"))
                }
    yield datasets

class TestTraining(TestESN):

    def test_simple(self, test_data):
        """where input = output, no other options"""
        esn = ESN(**self.kw)
        esn.build()
        esn.train(test_data[3]["u"])


    @pytest.mark.parametrize(
            "n_input",  (3, 7),
    )
    @pytest.mark.parametrize(
            "n_spinup", [0, 10],
    )
    @pytest.mark.parametrize(
            "batch_size", [None, 33, 10_000],
    )
    def test_all_options(self, test_data, n_input, n_spinup, batch_size):
        expected = test_data[n_input]

        kwargs = self.kw.copy()
        kwargs["n_input"] = n_input
        kwargs["n_output"] = n_input
        esn = ESN(**kwargs)
        esn.build()
        esn.train(expected["u"], y=expected["y"], n_spinup=n_spinup, batch_size=batch_size)

        assert esn.Wout.shape == (self.n_output, self.n_reservoir)

    def test_spinup_assert(self, test_data):
        esn = ESN(**self.kw)
        esn.build()
        with pytest.raises(AssertionError):
            esn.train(test_data[self.n_input]["u"], n_spinup=self.n_train+1)


    def test_time_is_last(self, test_data):
        esn = ESN(**self.kw)
        esn.build()
        with pytest.raises(AssertionError):
            esn.train(test_data[self.n_input]["u"].T, n_spinup=0)


class TestPrediction(TestESN):
    n_steps = 10
    path    = "test-store.zarr"

    def custom_setup_method(self, test_data):
        u = test_data[self.n_input]["u"]
        esn = ESN(input_kwargs={"random_seed": 10}, adjacency_kwargs={"random_seed": 11}, bias_kwargs={"random_seed": 12}, **self.kw)
        esn.build()
        esn.train(u)
        return esn, u

    def test_simple(self, test_data):
        """where input = output, no other options"""
        esn, u = self.custom_setup_method(test_data)

        v = esn.predict(u, n_steps=self.n_steps, n_spinup=0)

        # With zero spinup, these arrays actually should be equal
        assert_array_equal(v[:, 0], u[:, 0])
        assert v.shape == (esn.n_output, self.n_steps+1)

    @pytest.mark.parametrize(
            "n_spinup", (0, 10, 100_000)
    )
    def test_all_options(self, test_data, n_spinup):
        esn, u = self.custom_setup_method(test_data)

        if n_spinup > u.shape[-1]:
            with pytest.raises(AssertionError):
                v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)
        else:
            v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)

            assert v.shape == (esn.n_output, self.n_steps+1)


    @pytest.mark.parametrize(
            "n_spinup", (0, 10, 100_000)
    )
    def test_testing(self, test_data, n_spinup):
        esn, u = self.custom_setup_method(test_data)

        if n_spinup > u.shape[-1]:
            with pytest.raises(AssertionError):
                xds = esn.test(u, n_steps=self.n_steps, n_spinup=n_spinup)
        else:
            xds = esn.test(u, n_steps=self.n_steps, n_spinup=n_spinup)

            assert xds["prediction"].shape == (esn.n_output, self.n_steps+1)
            assert xds["prediction"].shape == xds["truth"].shape
            assert xds["prediction"].dims == xds["truth"].dims
            assert_array_equal(xds["prediction"].isel(ftime=0), xds["truth"].isel(ftime=0))


    def test_time_is_last(self, test_data):
        esn, u = self.custom_setup_method(test_data)

        with pytest.raises(AssertionError):
            esn.predict(u.T, n_steps=self.n_steps, n_spinup=0)


    def test_storage(self, test_data):
        esn, u = self.custom_setup_method(test_data)
        ds = esn.to_xds()

        # Make sure dataset matches
        for key in self.kw.keys():

            expected = getattr(esn, key)
            test = ds.attrs[key]

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

        ds.to_zarr(self.path, mode="w")
        esn2 = from_zarr(self.path)
        for key in self.kw.keys():

            expected = getattr(esn, key)
            test = getattr(esn2, key)

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

        for key in ["Win", "bias_vector", "Wout"]:
            assert_allclose(getattr(esn, key), getattr(esn2, key))

        assert_allclose(esn.W.data, esn2.W.data)

        # make sure Wout is a numpy array
        assert isinstance(esn2.Wout, np.ndarray)

        v1 = esn.predict(u, n_steps=self.n_steps, n_spinup=1)
        v2= esn2.predict(u, n_steps=self.n_steps, n_spinup=1)
        assert_allclose(v1, v2)

        rmtree(self.path)


    def test_storage_no_wout(self, test_data):
        esn, u = self.custom_setup_method(test_data)
        ds = esn.to_xds()
        del ds["Wout"]
        ds.to_zarr(self.path, mode="w")

        with pytest.warns():
            esn = from_zarr(self.path)

        rmtree(self.path)


    def test_no_Wout(self):
        esn = ESN(**self.kw)
        with pytest.raises(Exception):
            esn.to_xds()
