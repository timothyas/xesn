import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

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

    def test_errors(self):

        kw = self.kw.copy()
        kw["overlap"] = {0:1, 1:1, 2:1, 3:0}
        kw["data_chunks"] = (2, 2, 2, 1000)
        with pytest.raises(NotImplementedError):
            esn = LazyESN(**kw)

