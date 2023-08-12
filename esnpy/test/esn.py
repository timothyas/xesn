import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

from esnpy.esn import ESN
from esnpy.io import from_zarr

class TestESN:
    n_input             = 3
    n_output            = 3
    n_reservoir         = 100
    connectedness       = 5
    bias                = 0.1
    leak_rate           = 0.5
    tikhonov_parameter  = 1e-6

    input_factor        = 0.1
    adjacency_factor    = 0.1

    @property
    def kw(self):
        return {key: getattr(self, key) for key in [
            "n_input", "n_output", "n_reservoir", "connectedness", "bias", "leak_rate", "tikhonov_parameter", "input_factor", "adjacency_factor"]}


class TestInit(TestESN):

    def test_basic(self):
        esn = ESN(**self.kw)
        str(esn)
        assert esn.__repr__() == str(esn)

        for key in ["n_input", "n_output", "n_reservoir"]:
            expected = self.kw[key]
            test = getattr(esn, key)
            assert test == expected

        for key in ["input_factor", "adjacency_factor", "connectedness", "bias", "leak_rate", "tikhonov_parameter"]:
            expected = self.kw[key]
            test = getattr(esn, key)
            assert_allclose(test, expected)


    @pytest.mark.parametrize(
            "key, val, raises, error",
            [
                ("bias", -1., pytest.raises, ValueError),
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


class TestTraining(TestESN):
    n_train = 500
    rs      = np.random.RandomState(0)

    def test_simple(self):
        """where input = output, no other options"""
        u = self.rs.normal(size=(self.n_input, self.n_train))
        esn = ESN(**self.kw)
        esn.build()
        esn.train(u)


    @pytest.mark.parametrize(
            "n_input, n_output",  [(3, 7), (3, 3)],
    )
    @pytest.mark.parametrize(
            "n_spinup", [0, 10],
    )
    @pytest.mark.parametrize(
            "batch_size", [None, 33, 10_000],
    )
    def test_all_options(self, n_input, n_output, n_spinup, batch_size):
        u = self.rs.normal(size=(n_input, self.n_train))
        y = self.rs.normal(size=(n_output, self.n_train))

        kwargs = self.kw.copy()
        kwargs["n_input"] = n_input
        kwargs["n_output"] = n_input
        esn = ESN(**kwargs)
        esn.build()
        esn.train(u, y=y, n_spinup=n_spinup, batch_size=batch_size)

        assert esn.Wout.shape == (n_output, self.n_reservoir)

    def test_spinup_assert(self):
        u = self.rs.normal(size=(self.n_input, self.n_train))
        esn = ESN(**self.kw)
        esn.build()
        with pytest.raises(AssertionError):
            esn.train(u, n_spinup=self.n_train+1)


class TestPrediction(TestESN):
    n_train = 500
    n_steps = 10
    rs      = np.random.RandomState(0)
    path    = "test-store.zarr"

    def setup_method(self):
        u = self.rs.normal(size=(self.n_input, self.n_train))
        esn = ESN(input_kwargs={"random_seed": 10}, adjacency_kwargs={"random_seed": 11}, bias_kwargs={"random_seed": 12}, **self.kw)
        esn.build()
        esn.train(u)
        return esn, u

    def test_simple(self):
        """where input = output, no other options"""
        esn, u = self.setup_method()

        v = esn.predict(u, n_steps=self.n_steps, n_spinup=0)

        # With zero spinup, these arrays actually should be equal
        assert_array_equal(v[:, 0], u[:, 0])
        assert v.shape == (esn.n_output, self.n_steps+1)

    @pytest.mark.parametrize(
            "n_spinup", (0, 10, 100_000)
    )
    def test_all_options(self, n_spinup):
        esn, u = self.setup_method()

        if n_spinup > u.shape[-1]:
            with pytest.raises(AssertionError):
                v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)
        else:
            v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)

            assert v.shape == (esn.n_output, self.n_steps+1)


    def test_storage(self):
        esn, u = self.setup_method()
        ds = esn.to_xds()

        # Make sure dataset matches
        for key, expected in self.kw.items():
            assert_allclose(ds.attrs[key], expected)

        ds.to_zarr(self.path, mode="w")
        esn2 = from_zarr(self.path)
        for key in self.kw.keys():
            assert_allclose(getattr(esn, key), getattr(esn2, key))

        for key in ["Win", "bias_vector", "Wout"]:
            assert_allclose(getattr(esn, key), getattr(esn2, key))

        assert_allclose(esn.W.data, esn2.W.data)

        v1 = esn.predict(u, n_steps=self.n_steps, n_spinup=1)
        v2= esn2.predict(u, n_steps=self.n_steps, n_spinup=1)
        assert_allclose(v1, v2)

        rmtree(self.path)

    def test_no_Wout(self):
        esn = ESN(**self.kw)
        with pytest.raises(Exception):
            esn.to_xds()