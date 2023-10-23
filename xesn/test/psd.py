import pytest

import numpy as np

from xesn.test.xdata import test_data
from xesn.psd import psd

@pytest.mark.parametrize(
        "isel", ({}, {"z":0}, {"y":0, "z":0})
    )
def test_basic(test_data, isel):
    xda = psd(test_data.isel(**isel))
    assert xda.dims == ("k1d", "time")
    assert xda.shape == (len(test_data.x)//2, len(test_data.time))

def test_timefirst(test_data):
    with pytest.raises(AssertionError):
        psd(test_data.transpose("time", "z", "y", "x"))

def test_2d_dimlimit(test_data):
    xx = test_data.expand_dims({"m": np.arange(3)})
    with pytest.raises(NotImplementedError):
        psd(xx)
