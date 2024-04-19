import pytest

from copy import deepcopy
import numpy as np
import xarray as xr
from shutil import rmtree

from xesn import _use_cupy
from xesn.esn import ESN
from xesn.io import from_zarr

if _use_cupy:
    from cupy.testing import assert_allclose, assert_array_equal
    import cupy as xp
    import cupy_xarray
else:
    from numpy.testing import assert_allclose, assert_array_equal
    import numpy as xp

class TestESN:
    n_input             = 3
    n_output            = 3
    n_reservoir         = 100
    n_train             = 500
    leak_rate           = 0.5
    tikhonov_parameter  = 1e-6

    input_kwargs = {
            "factor"    : 0.1,
            "distribution": "uniform",
            "random_seed": 10,
            }
    adjacency_kwargs = {
            "factor"    : 0.1,
            "distribution": "uniform",
            "is_sparse": True,
            "connectedness": 5,
            "random_seed": 11,
            }
    bias_kwargs = {
            "factor"    : 0.1,
            "distribution": "uniform",
            "random_seed": 12,
            }


    @property
    def kw(self):
        return {key: getattr(self, key) for key in [
            "n_input", "n_output", "n_reservoir", "leak_rate", "tikhonov_parameter", "input_kwargs", "adjacency_kwargs", "bias_kwargs"]}

    equal_list  = ("n_input", "n_output", "n_reservoir")
    close_list  = ("leak_rate", "tikhonov_parameter")


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

        # check properties
        assert_allclose(esn.input_factor, self.input_kwargs["factor"])
        assert_allclose(esn.adjacency_factor, self.adjacency_kwargs["factor"])
        assert_allclose(esn.bias_factor, self.bias_kwargs["factor"])

    @pytest.mark.parametrize(
            "key, val, raises, error",
            [
                ("bias_kwargs", {"factor": -1.}, pytest.raises, ValueError),
                ("adjacency_kwargs", {"connectedness": 101}, pytest.raises, ValueError),
                ("adjacency_kwargs", {"connectedness": 90}, pytest.warns, RuntimeWarning),
            ],
    )
    def test_errors(self, key, val, raises, error):
        kwargs = deepcopy(self.kw)
        kwargs[key].update(val)

        with raises(error):
            esn = ESN(**kwargs)
            esn.build()


class TestVectors(TestESN):

    def test_bias_default(self):
        kw = deepcopy(self.kw)
        kw.pop("bias_kwargs")
        with pytest.warns():
            esn = ESN(**kw)

        esn.build()
        assert esn.bias_vector.shape == (self.n_reservoir,)

    @pytest.mark.parametrize(
            "distribution", ["uniform", "normal", "gaussian"],
    )
    def test_bias_kwargs(self, distribution):

        kw = deepcopy(self.kw)
        kw["bias_kwargs"] ={ "distribution"  : distribution,
                "factor"        : 0.1,
                "random_seed"   : 0,
                }
        esn = ESN(**kw)
        for key, expected in kw["bias_kwargs"].items():
            if isinstance(expected, float):
                assert_allclose(esn.bias_kwargs[key], expected)
            else:
                assert esn.bias_kwargs[key] == expected

        esn.build()
        assert esn.bias_vector.shape == (self.n_reservoir,)

    @pytest.mark.parametrize(
            "option", [
                {"normalization": "svd"},
                {"is_sparse": True},
                {"density": 0.1},
                {"sparsity": 0.9},
                {"connectedness": 1},
                {"format": "coo"},
            ]
        )
    def test_bias_badkwargs(self, option):
        kw = deepcopy(self.kw)
        kw["bias_kwargs"].update(option)
        with pytest.warns():
            ESN(**kw)


class TestMatricesDefault(TestESN):

    def test_input_default(self):
        kw = deepcopy(self.kw)
        kw.pop("input_kwargs")
        with pytest.warns():
            esn = ESN(**kw)

        esn.build()
        assert esn.Win.shape == (self.n_reservoir, self.n_input)



@pytest.mark.parametrize(
        "distribution", ["uniform", "normal", "gaussian"],
)
class TestMatrices(TestESN):

    @pytest.mark.parametrize(
            "is_sparse, sparse_key, sparse_arg, error", [
                (True, "density", 0.1, None),
                (True, "sparsity", 0.9, None),
                (True, "connectedness", 10, TypeError),
                (False, None, None, None)
            ],
    )
    @pytest.mark.parametrize(
            "normalization", ["multiply", "svd"],
    )
    def test_input_kwargs(self, distribution, normalization, is_sparse, sparse_key, sparse_arg, error):
        ikw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : 0.1,
                "random_seed"   : 0,
                }
        if is_sparse:
            ikw[sparse_key] = sparse_arg

        kw = deepcopy(self.kw)
        kw["input_kwargs"] = ikw.copy()
        if error is None:
            esn = ESN(**kw)
            for key, expected in ikw.items():
                assert esn.input_kwargs[key] == expected

            esn.build()
            assert tuple(esn.Win.shape) == (self.n_reservoir, self.n_input)
        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, input_kwargs=ikw)


    @pytest.mark.parametrize(
            "is_sparse, sparse_key, sparse_arg, error", [
                (True, "density", 0.1, None),
                (True, "sparsity", 0.9, None),
                (True, "connectedness", 10, None),
                (False, None, None, None)
            ],
    )
    @pytest.mark.parametrize(
            "normalization", ["multiply"],#, "eig", "svd"],
    )
    def test_adjacency_kwargs(self, distribution, normalization, is_sparse, sparse_key, sparse_arg, error):
        akw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : 0.1,
                "random_seed"   : 0,
                }
        if is_sparse:
            akw[sparse_key] = sparse_arg

        kw = deepcopy(self.kw)
        kw["adjacency_kwargs"] = akw.copy()
        if error is None:
            esn = ESN(**kw)
            for key, expected in akw.items():
                assert esn.adjacency_kwargs[key] == expected

            esn.build()
            assert tuple(esn.W.shape) == (self.n_reservoir, self.n_reservoir)
        else:
            with pytest.raises(error):
                esn = ESN(**kw)


@pytest.fixture(scope="module")
def test_data():
    rs = np.random.RandomState(0)
    tester = TestESN()

    datasets = {}

    time = np.arange(tester.n_train).astype(float)
    time = xr.DataArray(
        time,
        coords={"time": time},
        dims=("time",),
        attrs={"delta_t": 1., "units": "hours"},
    )

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
        if _use_cupy:
            datasets[n_input]["u"] = datasets[n_input]["u"].as_cupy()
            datasets[n_input]["y"] = datasets[n_input]["y"].as_cupy()

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

        kwargs = deepcopy(self.kw)
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
        esn = ESN(**self.kw)
        esn.build()
        esn.train(u)
        return esn, u

    def test_simple(self, test_data):
        """where input = output, no other options"""
        esn, u = self.custom_setup_method(test_data)

        v = esn.predict(u, n_steps=self.n_steps, n_spinup=0)

        # With zero spinup, these arrays actually should be equal
        assert_array_equal(v[:, 0].data, u[:, 0].data)
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
            assert_array_equal(xds["prediction"].isel(ftime=0).data, xds["truth"].isel(ftime=0).data)


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

        if _use_cupy:
            ds = ds.as_numpy()
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

        # make sure Wout is a numpy or cupy (not dask) array
        assert isinstance(esn2.Wout, xp.ndarray)

        v1 = esn.predict(u, n_steps=self.n_steps, n_spinup=1)
        v2= esn2.predict(u, n_steps=self.n_steps, n_spinup=1)
        assert_allclose(v1.data, v2.data)

        rmtree(self.path)


    def test_storage_no_wout(self, test_data):
        esn, u = self.custom_setup_method(test_data)
        ds = esn.to_xds()
        del ds["Wout"]
        if _use_cupy:
            ds = ds.as_numpy()
        ds.to_zarr(self.path, mode="w")

        with pytest.warns():
            esn = from_zarr(self.path)

        rmtree(self.path)


    def test_no_Wout(self):
        esn = ESN(**self.kw)
        with pytest.raises(Exception):
            esn.to_xds()
