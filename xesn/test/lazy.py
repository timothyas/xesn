import pytest

import numpy as np

import xarray as xr
from shutil import rmtree

import dask.array as darray

from xesn import _use_cupy
from xesn.lazyesn import LazyESN
from xesn.io import from_zarr

from xesn.test.esn import TestESN

if _use_cupy:
    from cupy.testing import assert_allclose, assert_array_equal
    import cupy_xarray
else:
    from numpy.testing import assert_allclose, assert_array_equal


class TestLazy(TestESN):

    n_input     = 5
    n_output    = 3
    n_train     = 500
    esn_chunks  = {"x": 3, "time": 1_000}
    overlap     = {"x": 1, "time": 0}
    boundary    = 0
    persist     = True
    equal_list  = ("overlap", "esn_chunks", "persist", "overlap", "n_reservoir", "boundary")
    close_list  = ("leak_rate", "tikhonov_parameter")

    @property
    def kw(self):
        keys = ["esn_chunks", "overlap", "persist", "boundary"]
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
    # zchunks is used to create the data, mirroring a zarr store with smaller chunks
    # than what we want for the ESN
    for dims, shape, zchunks, chunks, overlap, Wout_chunks, Wout_shape in zip(
            ( ("x",),     ("x", "y"),   ("x", "y", "z") ),
            ( ( 4 ,),     ( 4 ,  6 ),   ( 4 ,  6 ,  5 ) ),
            ( ( 1 ,),     ( 1 ,  3 ),   ( 1 ,  3 ,  5 ) ),
            ( ( 2 ,),     ( 2 ,  3 ),   ( 2 ,  3 ,  5 ) ),
            ( ( 1 ,),     ( 1 ,  1 ),   ( 1 ,  1 ,  0 ) ),
            ( [2, None], [  6, None], [   30, None, 1]),
            ( [2, None], [2*6, None], [2*6*5, None, 1]),
            ):
        shape += (tester.n_train,)
        nd = len(shape)
        dims += ("time",)
        zchunks += (tester.n_train/5,)
        chunks += (-1,)
        overlap += (0,)

        Wout_chunks[1] = tester.n_reservoir
        Wout_shape[1]  = tester.n_reservoir*2

        data = xr.DataArray(
            rs.normal(size=shape, chunks=zchunks),
            dims=dims
        )
        if _use_cupy:
            data = data.as_cupy()
        datasets[nd] = {
                "data": data,
                "shape": shape,
                "chunks": dict(zip(dims, chunks)),
                "overlap": dict(zip(dims, overlap)),
                "Wout_chunks": Wout_chunks,
                "Wout_shape": Wout_shape,
                "overlap": {d:o for d, o in zip(dims,overlap)},
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
        assert esn.esn_chunks == esn.output_chunks
        assert esn.input_chunks == {"x":self.n_input, "time": self.esn_chunks["time"]}
        assert esn.ndim_state == 1
        assert esn._r_chunks == (self.n_reservoir,)
        assert esn._Wout_chunks == (self.n_output, self.n_reservoir)


    @pytest.mark.parametrize(
            "attr, expected", [
                ("overlap", 0),
                ("esn_chunks", -1)
            ]
        )
    def test_default_time(self, attr, expected):

        kw = self.kw.copy()
        kw[attr].pop("time")

        esn = LazyESN(**kw)
        assert getattr(esn, attr)["time"] == expected


    def test_not_implemented(self):

        kw = self.kw.copy()
        kw["overlap"] = {"x":1, "y":1, "z":1, "time":0}
        kw["esn_chunks"] = {"x":2, "y":2, "z":2, "time":1000}
        with pytest.raises(NotImplementedError):
            esn = LazyESN(**kw)

    def test_for_negative_chunksizes(self):

        kw = self.kw.copy()
        kw["esn_chunks"] = {"x":3, "y": -1, "time": 1_000}
        kw["overlap"] = {"x":3, "y":0, "time":1_000}
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

        # also test this form of boundary
        if n_dim == 4:
            kw["boundary"] = {"x": "periodic", "y": 0., "z": "reflect"}
        kw["esn_chunks"] = expected["chunks"]
        kw["overlap"] = expected["overlap"]

        esn = LazyESN(**kw)
        esn.build()
        esn.train(u, n_spinup=n_spinup, batch_size=batch_size)

        assert esn.Wout.chunksize == tuple(expected["Wout_chunks"])
        assert esn.Wout.shape == tuple(expected["Wout_shape"])

    def test_time_is_last(self, test_data):

        expected = test_data[2]

        u = expected["data"]
        kw = self.kw.copy()
        kw["esn_chunks"] = expected["chunks"]
        kw["overlap"] = expected["overlap"]

        esn = LazyESN(**kw)
        esn.build()
        with pytest.raises(AssertionError):
            esn.train(u.T, n_spinup=0, batch_size=100)



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
        kw["esn_chunks"] = chunks
        kw["overlap"] = overlap
        if len(chunks) == 4:
            kw["boundary"] = {"x": "periodic", "y": 0., "z": "reflect"}

        esn = LazyESN(**kw)
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
        assert_array_equal(v[..., 0].compute().data, u[..., 0].compute().data)
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

    @pytest.mark.parametrize(
            "n_spinup", (0, 10, 100_000)
    )
    def test_testing(self, test_data, n_dim, n_spinup):

        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)

        if n_spinup > u.shape[-1]:
            with pytest.raises(AssertionError):
                xds = esn.test(u, n_steps=self.n_steps, n_spinup=n_spinup)
        else:
            xds = esn.test(u, n_steps=self.n_steps, n_spinup=n_spinup)

            assert xds["prediction"].shape == expected["shape"][:-1] + (self.n_steps+1,)
            assert xds["prediction"].shape == xds["truth"].shape
            assert xds["prediction"].data.chunksize[:-1] == tuple(esn.output_chunks.values())[:-1]
            assert xds["prediction"].dims == xds["truth"].dims
            assert_array_equal(
                xds["prediction"].isel(ftime=0).compute().data,
                xds["truth"].isel(ftime=0).compute().data
            )

    def test_time_is_last(self, test_data, n_dim):
        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)

        with pytest.raises(AssertionError):
            esn.predict(u.T, n_steps=self.n_steps, n_spinup=0)


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
        if _use_cupy:
            ds = ds.compute().as_numpy()
            ds["Wout"] = ds["Wout"].chunk({"iy": esn.n_output, "ir": esn.n_reservoir})
        ds.to_zarr(self.path, mode="w")
        esn2 = from_zarr(self.path)
        for key in self.kw.keys():

            expected = getattr(esn, key)
            test = getattr(esn2, key)

            if key in self.equal_list:
                assert test == expected
            elif key in self.close_list:
                assert_allclose(test, expected)

        for key in ["Win", "bias_vector"]:
            assert_allclose(getattr(esn, key), getattr(esn2, key))

        assert_allclose(esn.Wout.compute(), esn2.Wout.compute())
        assert_allclose(esn.W.data, esn2.W.data)

        # make sure Wout is a dask array
        assert isinstance(esn2.Wout, darray.core.Array)

        v1 = esn.predict(u, n_steps=self.n_steps, n_spinup=1)
        v2= esn2.predict(u, n_steps=self.n_steps, n_spinup=1)
        assert_allclose(v1.data.compute(), v2.data.compute())

        rmtree(self.path)


    def test_storage_no_wout(self, test_data, n_dim):
        expected = test_data[n_dim]
        esn = self.custom_setup_method(expected["chunks"], expected["overlap"])

        u = expected["data"]
        esn.train(u)
        ds = esn.to_xds()
        del ds["Wout"]
        ds.to_zarr(self.path, mode="w")

        with pytest.warns():
            esn = from_zarr(self.path)

        rmtree(self.path)
