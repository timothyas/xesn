import pytest

import numpy as np
import xarray as xr
from shutil import rmtree

import dask.array as darray

from xesn.xdata import XData
from xesn import _use_cupy
if _use_cupy:
    from cupy.testing import assert_allclose, assert_array_equal
else:
    from numpy.testing import assert_allclose, assert_array_equal


class TestXData:

    field_name  = "theta"
    zstore_path = "test-xdata.zarr"
    dimensions  = ("x", "y", "z", "time")

    @property
    def subsampling(self):

        s = {"x"    : (None, 4, None),
             "y"    : (None, 5, 1),
             "z"    : (1, None, 2),
             "time" : {
                 "training"     : (None, 100, 2),
                 "validation"   : (100, 150, None),
                 "testing"      : (150, 200, None)},
             }
        return s

    @property
    def value_sampling(self):

        s = {"x"    : (-5.,  5., None),
             "y"    : (-4.,  4., None),
             "z"    : (10., 30., None),
             "time" : {
                 "training"     : (None, 1000., None),
                 "validation"   : (1000., 1500., None),
                 "testing"      : (1500., 2000., None)},
             }
        return s

    @property
    def normalization(self):
        n = {"bias": 3.,
             "scale": 4}
        return n

    @property
    def kw(self):
        keys = ["field_name", "zstore_path", "subsampling", "normalization"]
        return {key: getattr(self, key) for key in keys}


@pytest.fixture(scope="module")
def test_data():

    rs = darray.random.RandomState(0)
    tester = TestXData()
    u = xr.DataArray(
            rs.normal(size=(10,10,5,200), chunks=(5,5,5,200)),
            coords={
                "x": np.linspace(-10, 10, 10),
                "y": np.linspace(-10, 10, 10),
                "z": np.linspace(  0, 50, 5),
                "time": np.linspace(0, 2000, 200),
                },
            dims=tester.dimensions,
            attrs={"description": "This is some test data!"},
            )
    u.name = tester.field_name
    u.to_dataset().to_zarr(tester.zstore_path, mode="w")
    yield u
    rmtree(tester.zstore_path)


class TestInit(TestXData):

    def test_minimum(self):
        xd = XData(self.field_name, self.zstore_path)
        for key in ["field_name", "zstore_path"]:
            assert getattr(xd, key) == getattr(self, key)

        for key in ["dimensions", "subsampling", "normalization"]:
            assert getattr(xd, key) is None


    def test_extras(self):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   subsampling=self.subsampling,
                   normalization=self.normalization)
        for key in ["field_name", "zstore_path", "dimensions", "subsampling", "normalization"]:
            assert getattr(xd, key) == getattr(self, key)

class TestSetup(TestXData):


    def sampler(self, test_data, dim, mode, style):
        if style == "index":
            slc = slice(*self.subsampling[dim]) if dim != "time" else slice(*self.subsampling[dim][mode])
            expected = test_data.isel({dim: slc})
        else:
            slc = slice(*self.value_sampling[dim]) if dim != "time" else slice(*self.value_sampling[dim][mode])
            expected = test_data.sel({dim: slc})

        if _use_cupy:
            expected = expected.as_cupy()
        return expected

    def test_pass(self, test_data):
        """make sure when nothing is specified, nothing happens"""
        xd = XData(self.field_name, self.zstore_path)
        test = xd.subsample(test_data, mode="training")
        assert_array_equal(test, test_data)

        test = xd.normalize(test_data)
        assert_array_equal(test, test_data)



    @pytest.mark.parametrize(
            "mode", ("training", "validation", "testing"),
        )
    @pytest.mark.parametrize(
            "style", ("index", "value",),
        )
    def test_subsample(self, test_data, mode, style):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   subsampling=self.subsampling if style=="index" else self.value_sampling,
                   normalization=self.normalization)

        test = xd.subsample(test_data, mode=mode)
        for dim in self.dimensions:
            expected = self.sampler(test_data, dim, mode, style)
            assert_allclose(test[dim], expected[dim])


    def test_subsample_error(self, test_data):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   # below is the error, mix ints and floats
                   subsampling={"x": (0, 10.),
                                "y": (-2, 2.),
                                "z": (0,30.),
                                "time": {
                                    "training": (0, 100.),
                                    "validation": (100., 150),
                                    "testing": (150., 200),
                                    },
                                }
                   )
        with pytest.raises(TypeError):
            xd.subsample(test_data, mode="training")


    def test_subsample_type(self, test_data):

        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   subsampling={"time":{"training":(np.datetime64("2000-01-01"),)}})

        with pytest.raises(TypeError):
            xd.subsample(test_data, mode="training")


    @pytest.mark.parametrize(
            "keep_attrs", (True, False)
        )
    def test_normalize(self, test_data, keep_attrs):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   normalization=self.normalization)

        test = xd.normalize(test_data, keep_attrs=keep_attrs)
        expected = (test_data - self.normalization["bias"]) / self.normalization["scale"]
        assert_allclose(test, expected)

        if keep_attrs:
            assert test.attrs == test_data.attrs


    @pytest.mark.parametrize(
            "keep_attrs", (True, False)
        )
    def test_normalize_inverse(self, test_data, keep_attrs):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   normalization=self.normalization)

        test = xd.normalize_inverse(test_data, keep_attrs)
        expected = test_data *  self.normalization["scale"] + self.normalization["bias"]
        assert_allclose(test, expected)

        if keep_attrs:
            assert test.attrs == test_data.attrs


    # Some repetition here, but I think it's worth it
    @pytest.mark.parametrize(
            "mode", ("training", "validation", "testing"),
        )
    @pytest.mark.parametrize(
            "style", ("index", "value",),
        )
    def test_setup(self, test_data, mode, style):
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=self.dimensions,
                   subsampling=self.subsampling if style=="index" else self.value_sampling,
                   normalization=self.normalization)

        test = xd.setup(mode=mode)
        expected = test_data.copy()
        for dim in self.dimensions:
            expected = self.sampler(expected, dim, mode, style)
            assert_allclose(test[dim], expected[dim])

        expected = (expected - self.normalization["bias"]) / self.normalization["scale"]
        assert_allclose(test.compute().data, expected.compute().data)

    def test_transpose(self, test_data):
        dimsT = ("x","z","y","time")
        xd = XData(self.field_name,
                   self.zstore_path,
                   dimensions=dimsT,
                   subsampling=self.subsampling,
                   normalization=self.normalization)


        test = xd.setup(mode="training")
        assert test.dims == dimsT
