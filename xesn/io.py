
import inspect
import warnings
import xarray as xr

from . import _use_cupy
from .esn import ESN
from .lazyesn import LazyESN

if _use_cupy:
    import cupy_xarray

def from_zarr(store, **kwargs):
    """Create an ESN or LazyESN from a zarr store.

    Args:
        store (MutableMapping or str): path or mapping to a filesystem with a zarr store. See xarray.open_zarr for details.
        **kwargs (dict): additional keyword arguments are passed to xarray.open_zarr

    Returns:
        esn (ESN or LazyESN): an ESN or distributed ESN
    """

    xds = xr.open_zarr(store, **kwargs)

    # Create ESN
    is_lazy = "overlap" in xds.attrs
    ESNModel = LazyESN if is_lazy else ESN

    # Only get __init__ arguments from dataset attributes
    kw, *_ = inspect.getfullargspec(ESNModel.__init__)
    kw.remove("self")
    args = {key: xds.attrs[key] for key in kw}

    esn = ESNModel(**args)
    esn.build()

    if "Wout" in xds:
        if _use_cupy:
            Wout = xds["Wout"].as_cupy().data
            Wout = Wout if is_lazy else Wout.compute()
        else:
            Wout = xds["Wout"].data if is_lazy else xds["Wout"].values

        # Need to re-append singleton dimension
        if is_lazy:
            for _ in range(esn.ndim_state - 2):
                Wout = Wout[...,None]

        esn.Wout = Wout
    else:
        warnings.warn(f"from_zarr: did not find 'Wout' in zarr store, returning an untrained network.")
    return esn
