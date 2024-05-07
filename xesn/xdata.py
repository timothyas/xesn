

import numpy as np
import xarray as xr


from . import _use_cupy

if _use_cupy:
    import cupy_xarray

class XData():
    """A class for very simple processing routines for xarray.DataArrays.
    See :meth:`setup` for the main usage.
    """

    __slots__ = (
        "field_name", "zstore_path",
        "dimensions", "subsampling", "normalization",
    )

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
        """Main data processing routine, consisting of the following steps:

        1. Open a dataset with xarray.open_zarr, and select the field
        2. Potentially transpose the data to the order of :attr:`dimensions`
        3. Use :meth:`subsample` to select the data, based on the ``mode`` argument
        4. Use :meth:`normalize` to perform very simple normalization

        Args:
            mode (str): either "training", "validation", or "testing"

        Returns:
            xda (xarray.Dataset): containing the desired field with any preprocessing done
        """

        # get lazy data, return xarray dataset
        ds = xr.open_zarr(self.zstore_path)
        xda = ds[self.field_name]

        dims = self.dimensions if self.dimensions is not None else xda.dims
        if tuple(dims) != xda.dims:
            xda = xda.transpose(*dims)

        xda = self.subsample(xda, mode=mode)
        xda = self.normalize(xda)

        if _use_cupy:
            xda = xda.as_cupy()

        # TODO: other preprocessing, like adding noise
        return xda


    def subsample(self, xda, mode):
        """Subsample a DataArray along axes specified in :attr:`subsampling`.
        Note that the "time" section has to have modes, e.g., indicating training, validation, or testing.

        Note:
            Each dimension can be subsampled either via indices if integers are provided,
            or by value if floats are provided. These types can't be mixed for the same axis though, and
            no other type is supported, e.g., numpy.datetime64 or datetime.datetime, because
            these types can't be specified easily in a yaml file to the Driver class, which
            is the main point of interaction here.

        Args:
            xda (xarray.DataArray): a multi-dimensional array to be subsampled
            mode (str): indicating how to subsample the time dimension

        Returns:
            xda (xarray.DataArray): subsampled
        """

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

            elif not (myslice.start is None and myslice.stop is None):
                raise TypeError(f"XData.subsample: unrecognized slice type, only ints or floats allowed")

            print(f"XData.subsample: Subsampled/sliced {dim}: {xda[dim]}")

        # Squeeze out any singular dimensions after slicing
        xda = xda.squeeze()
        return xda


    def normalize(self, xda, keep_attrs=False):
        """Very simple, this may eventually be hooked up with e.g., scikit-learn for more advanced normalization.
        Right now, normalize with scalars :attr:`bias` and :attr:`scale` as

        .. math::
            (xda - bias) / scale

        Args:
            xda (xarray.DataArray): with the field to be normalized
            keep_attrs (bool): if True, keep attributes in xda

        Returns:
            xda (xarray.DataArray): normalized
        """

        if self.normalization is None:
            return xda

        bias = self.normalization.get("bias", 0.)
        scale= self.normalization.get("scale", 1.)

        with xr.set_options(keep_attrs=keep_attrs):
            result = (xda - bias)/scale

        return result


    def normalize_inverse(self, xda, keep_attrs=False):
        """Do the inverse operation of :meth:`normalize`, i.e.,

        .. math::
            xda * scale + bias

        Args:
            xda (xarray.DataArray): with the normalized field to be re scaled
            keep_attrs (bool): if True, keep attributes in xda

        Returns:
            xda (xarray.DataArray): scaled and biased like original data
        """
        if self.normalization is None:
            return xda

        bias = self.normalization.get("bias", 0.)
        scale= self.normalization.get("scale", 1.)

        with xr.set_options(keep_attrs=keep_attrs):
            result = xda*scale + bias

        return result
