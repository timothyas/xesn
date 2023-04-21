
import xarray as xr

from .esn import ESN
from .lazyesn import LazyESN

def from_zarr(store, **kwargs):

    xds = xr.open_zarr(store, **kwargs)

    # Use dataset attributes to get __init__ arguments
    args = {}
    for key, val in xds.attrs.items():

        v = val
        if isinstance(val, str):
            if val.lower() == "none":
                v = None
            elif val.lower() in ("true", "false"):
                v = val.lower()[0] == "t"

        args[key] = v

    # Create ESN
    esn = ESN(**args) if "overlap" not in args else LazyESN(**args)
    esn.build()
    esn.Wout = xds["Wout"].data

    # Overlap dictionary is interpreted with string keys
    if "overlap" in args:
        esn.overlap = {int(k):v for k,v in esn.overlap.items()}

    return esn
