
import xarray as xr

from .esn import ESN
from .lazyesn import LazyESN

def from_zarr(store, **kwargs):

    xds = xr.open_zarr(store, **kwargs)

    # Use dataset attributes to get __init__ arguments
    args = {key: val for key, val in xds.attrs.items()}

    # Create ESN
    is_lazy = "overlap" in args
    esn = LazyESN(**args) if is_lazy else ESN(**args)
    esn.build()
    Wout = xds["Wout"].data if is_lazy else xds["Wout"].values

    # Need to re-append singleton dimension
    if is_lazy:
        for _ in range(esn.ndim_state - 2):
            Wout = Wout[...,None]

    esn.Wout = Wout
    return esn
