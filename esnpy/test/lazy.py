import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

import dask.array as darray

from esnpy.lazyesn import LazyESN
from esnpy.io import from_zarr

from esnpy.test.esn import TestESN

class TestLazy(TestESN):

    n_input     = 5
    n_output    = 3
    n_train     = 500
    data_chunks = (3, 1_000)
    overlap     = {0: 1, 1: 0}
    persist     = True
    equal_list  = ("overlap", "data_chunks", "persist", "overlap", "n_reservoir", "boundary")
    close_list  = ("input_factor", "adjacency_factor", "connectedness", "bias", "leak_rate", "tikhonov_parameter")

    @property
    def kw(self):
        keys = ["data_chunks", "overlap", "persist"]
        kw = super().kw.copy()
        kw.update({
            key: getattr(self, key) for key in keys})
        for key in ["n_input", "n_output"]:
            kw.pop(key)
        return kw


@pytest.fixture(scope="module")
def test_data():

    tester = TestLazy()
    rs = darray.random.RandomState(0)
    datasets = {}
    for shape, chunks, overlap, Wout_chunks, Wout_shape in zip(
            ( ( 4 ,),     ( 4 ,  6 ),   ( 4 ,  6 ,  5 ) ),
            ( ( 2 ,),     ( 2 ,  3 ),   ( 2 ,  3 ,  5 ) ),
            ( ( 1 ,),     ( 1 ,  1 ),   ( 1 ,  1 ,  0 ) ),
            ( [2, None], [  6, None], [   30, None, 1]),
            ( [2, None], [2*6, None], [2*6*5, None, 1]),
            ):
        shape += (tester.n_train,)
        nd = len(shape)
        chunks += (-1,)
        overlap += (0,)

        Wout_chunks[1] = tester.n_reservoir
        Wout_shape[1]  = tester.n_reservoir*2

        datasets[nd] = {
                "data": rs.normal(size=shape, chunks=chunks),
                "shape": shape,
                "chunks": chunks,
                "overlap": overlap,
                "Wout_chunks": Wout_chunks,
                "Wout_shape": Wout_shape,
                "overlap": {i:o for i, o in enumerate(overlap)},
                }
    yield datasets


class TestInit(TestLazy):

    def test_basic(self):

        esn = LazyESN(**self.kw)
        str(esn)
        assert esn.__repr__() == str(esn)

        for key in self.kw.keys():

            expected = getattr(self, key)
            test = getattr(esn, key)

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

        # test some basic properties to lock them in
        assert esn.data_chunks == esn.output_chunks
        assert esn.input_chunks == (self.n_input, self.data_chunks[-1])
        assert esn.ndim_state == 1
        assert esn.r_chunks == (self.n_reservoir,)
        assert esn.Wout_chunks == (self.n_output, self.n_reservoir)

    def test_not_implemented(self):

        kw = self.kw.copy()
        kw["overlap"] = {0:1, 1:1, 2:1, 3:0}
        kw["data_chunks"] = (2, 2, 2, 1000)
        with pytest.raises(NotImplementedError):
            esn = LazyESN(**kw)

    def test_for_negative_chunksizes(self):

        kw = self.kw.copy()
        kw["data_chunks"] = (3,-1,1_000)
        kw["overlap"] = (3, 0, 1_000)
        with pytest.raises(ValueError):
            esn = LazyESN(**kw)


class TestTraining(TestLazy):
    rs      = darray.random.RandomState(0)
    boundary= "periodic"

    @property
    def kw(self):
        kw = super().kw.copy()
        kw.update({key:getattr(self, key) for key in ["boundary"]})
        return kw


    @pytest.mark.parametrize(
            "n_dim", (2, 3, 4)
    )
    @pytest.mark.parametrize(
            "n_spinup", [0, 10],
    )
    @pytest.mark.parametrize(
            "batch_size", [None, 33, 10_000],
    )
    def test_many_sizes(self, test_data, n_dim, n_spinup, batch_size):

        expected = test_data[n_dim]

        u = expected["data"]
        kw = self.kw.copy()
        kw["data_chunks"] = expected["chunks"]
        kw["overlap"] = expected["overlap"]

        esn = LazyESN(**kw)
        esn.build()
        esn.train(u, n_spinup=n_spinup, batch_size=batch_size)

        assert esn.Wout.chunksize == tuple(expected["Wout_chunks"])
        assert esn.Wout.shape == tuple(expected["Wout_shape"])


# TODO: Data with NaNs...
@pytest.mark.parametrize(
        "n_dim", (2, 3, 4)
)
class TestPrediction(TestLazy):
    n_steps = 10
    boundary= "periodic"
    path    = "test-store.zarr"


    @property
    def kw(self):
        kw = super().kw.copy()
        kw.update({key:getattr(self, key) for key in ["boundary"]})
        return kw


    def custom_setup_method(self, chunks, overlap):
        kw = self.kw.copy()
        kw["data_chunks"] = chunks
        kw["overlap"] = overlap

        esn = LazyESN(
                input_kwargs={"random_seed": 10},
                adjacency_kwargs={"random_seed": 11},
                bias_kwargs={"random_seed": 12},
                **kw)
        esn.build()
        return esn


    def test_simple(self, test_data, n_dim):
        """where input = output, no other options"""

        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)
        v = esn.predict(u, n_steps=self.n_steps, n_spinup=0)

        # With zero spinup, these arrays actually should be equal
        assert_array_equal(v[..., 0], u[..., 0])
        assert v.shape == expected["shape"][:-1] + (self.n_steps+1,)


    @pytest.mark.parametrize(
            "n_spinup", (0, 10, 100_000)
    )
    def test_all_options(self, test_data, n_dim, n_spinup):

        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)

        if n_spinup > u.shape[-1]:
            with pytest.raises(AssertionError):
                v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)
        else:
            v = esn.predict(u, n_steps=self.n_steps, n_spinup=n_spinup)

            assert v.shape == expected["shape"][:-1] + (self.n_steps+1,)


    def test_storage(self, test_data, n_dim):
        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)
        ds = esn.to_xds()

        # Make sure dataset matches
        for key in self.kw.keys():

            expected = getattr(esn, key)
            test = ds.attrs[key]

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

        # Now store & read to make a second ESN
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

        # make sure Wout is a dask array
        assert isinstance(esn2.Wout, darray.core.Array)

        v1 = esn.predict(u, n_steps=self.n_steps, n_spinup=1)
        v2= esn2.predict(u, n_steps=self.n_steps, n_spinup=1)
        assert_allclose(v1, v2)

        rmtree(self.path)
