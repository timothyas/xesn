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
    data_chunks = (3, 1_000)
    overlap     = {0: 1, 1: 0}
    persist     = True

    @property
    def kw(self):
        keys = ["data_chunks", "overlap", "persist"]
        kw = super().kw.copy()
        kw.update({
            key: getattr(self, key) for key in keys})
        for key in ["n_input", "n_output"]:
            kw.pop(key)
        return kw


class TestInit(TestLazy):

    def test_basic(self):

        esn = LazyESN(**self.kw)
        str(esn)
        assert esn.__repr__() == str(esn)

        for key in ["n_input", "n_output", "overlap", "data_chunks", "persist", "overlap"]:
            expected = getattr(self, key)
            test = getattr(esn, key)
            assert test == expected

        for key in ["input_factor", "adjacency_factor", "connectedness", "bias", "leak_rate", "tikhonov_parameter"]:
            expected = self.kw[key]
            test = getattr(esn, key)
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
    n_train = 500
    rs      = darray.random.RandomState(0)
    boundary= "periodic"

    @property
    def kw(self):
        kw = super().kw.copy()
        kw.update({key:getattr(self, key) for key in ["boundary"]})
        return kw


    @pytest.mark.parametrize(
            "shape, chunks, overlap, Wout_chunks, Wout_shape", [
                ( (4,     ), (2,     ), (1,     ), [ 2, None   ], [  2,   None   ] ),
                ( (4, 6   ), (2, 3   ), (1, 1   ), [ 6, None   ], [2*6,   None   ] ),
                ( (4, 6, 5), (2, 3, 5), (1, 1, 0), [30, None, 1], [2*6*5, None, 1] ),
            ]
    )
    @pytest.mark.parametrize(
            "n_spinup", [0, 10],
    )
    @pytest.mark.parametrize(
            "batch_size", [None, 33, 10_000],
    )
    # Maybe better to separate the "options" testing from all the different sizes
    def test_many_sizes(self, shape, chunks, overlap, Wout_chunks, Wout_shape, n_spinup, batch_size):
        """where input = output, no other options"""
        shape += (self.n_train,)
        chunks += (-1,)
        overlap += (0,)

        Wout_chunks[1] = self.n_reservoir
        Wout_shape[1]  = self.n_reservoir*2

        kw = self.kw.copy()
        kw["data_chunks"] = chunks
        kw["overlap"] = {i:o for i, o in enumerate(overlap)}

        u = self.rs.normal(size=shape, chunks=chunks)
        esn = LazyESN(**kw)
        esn.build()
        esn.train(u, n_spinup=n_spinup, batch_size=batch_size)

        assert esn.Wout.chunksize == tuple(Wout_chunks)
        assert esn.Wout.shape == tuple(Wout_shape)


class TestPrediction(TestLazy):
    n_train = 500
    n_steps = 10
    rs      = darray.random.RandomState(0)
    boundary= "periodic"


# Data with NaNs...
