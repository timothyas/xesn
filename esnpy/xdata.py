

import numpy as np
import xarray as xr

class XData():

    def __init__(self,
                 field_name,
                 zstore_path,
                 dimensions=None,
                 subsampling=None,
                 normalization=None,
                 ):

        self.field_name     = field_name
        self.zstore_path    = zstore_path
        self.dimensions     = dimensions
        self.subsampling    = subsampling
        self.normalization  = normalization


    def setup(self, mode):
        """

        TODO:
        1. change the name
        2. how to print to logfile

        Args:
            mode (str): either "training", "validation", or "testing"

        Returns:
            xda (xarray.Dataset): containing the desired field with any preprocessing
        """

        # get lazy data, return xarray dataset
        ds = xr.open_zarr(self.zstore_path)
        xda = ds[self.field_name]

        dims = self.dimensions if self.dimensions is not None else xda.dims
        if tuple(dims) != xda.dims:
            xda = xda.transpose(*dims)

        xda = self.subsample(xda, mode=mode)
        xda = self.normalize(xda)

        # TODO: other preprocessing, like adding noise
        return xda


    def subsample(self, xda, mode):

        slices = self.subsampling if self.subsampling is not None else dict()
        for dim, slc in slices.items():

            # time is handled a bit differently
            tup = tuple(slc[mode]) if dim == "time" else tuple(slc)
            myslice = slice(*tup)

            # check for consistent start/stop dtypes
            try:
                assert type(myslice.start) == type(myslice.stop) or \
                       (myslice.start is None or myslice.stop is None)
            except:
                raise TypeError(f"XData.subsample: slice elements must be all ints or all floats. Got {myslice} for dim={dim}")

            # do the slicing
            print(f"XData.subsample: Original {dim}: {xda[dim]}")
            if isinstance(myslice.start, int) or isinstance(myslice.stop, int):
                xda = xda.isel({dim: myslice})

            elif isinstance(myslice.start, float) or isinstance(myslice.stop, float):
                xda = xda.sel({dim: myslice})

            print(f"XData.subsample: Subsampled/sliced {dim}: {xda[dim]}")

        # Squeeze out any singular dimensions after slicing
        xda = xda.squeeze()
        return xda


    def normalize(self, xda):

        # good practice?
        if self.normalization is None:
            return xda

        # TODO: not really clear how to handle other cases...
        # is it worth handling what options should be here?
        # assert self.normalization["type"] == "normal-scalar"
        bias = self.normalization.get("bias", 0.)
        scale= self.normalization.get("scale", 1.)
        return (xda - bias)/scale
