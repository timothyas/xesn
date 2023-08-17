
import xarray as xr

from .esn import ESN
from .lazyesn import LazyESN

def from_zarr(store, **kwargs):

    xds = xr.open_zarr(store, **kwargs)

    # Use dataset attributes to get __init__ arguments
    args = {key: val for key, val in xds.attrs.items()}

    # Create ESN
    esn = ESN(**args) if "overlap" not in args else LazyESN(**args)
    esn.build()
    esn.Wout = xds["Wout"].data

    # Overlap dictionary is interpreted with string keys
    if "overlap" in args:
        esn.overlap = {int(k):v for k,v in esn.overlap.items()}

    return esn
