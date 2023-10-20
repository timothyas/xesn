import pytest

import os
from os.path import join
from glob import glob
import yaml
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from shutil import rmtree

import dask.array as darray

from xesn.test.xdata import test_data
from xesn.psd import psd_1d, psd_2d

def test_1d(test_data):
    xda = psd_1d(test_data.isel(y=0, z=0))
    assert xda.dims == ("k1d", "time")
    assert xda.shape == (len(test_data.x)//2, len(test_data.time))


def test_1d_timefirst(test_data):
    with pytest.raises(AssertionError):
        psd_1d(test_data.isel(y=0, z=0).transpose("time", "x"))

def test_1d_multidim(test_data):
    with pytest.raises(AssertionError):
        psd_1d(test_data)

@pytest.mark.parametrize(
        "isel", ({}, {"z":0})
    )
def test_2d(test_data, isel):
    xda = psd_2d(test_data.isel(**isel))
    assert xda.dims == ("k1d", "time")
    assert xda.shape == (len(test_data.x)//2, len(test_data.time))


def test_2d_timefirst(test_data):
    with pytest.raises(AssertionError):
        psd_2d(test_data.transpose("time", "z", "y", "x"))

def test_2d_dimlimit(test_data):
    xx = test_data.expand_dims({"m": np.arange(3)})
    with pytest.raises(NotImplementedError):
        psd_2d(xx)
