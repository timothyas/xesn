# Figure out if we have cupy/GPU access
try:
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()
    _use_cupy = True

except ImportError:
    _use_cupy = False

except xp.cuda.runtime.CUDARuntimeError:
    _use_cupy = False

from .esn import ESN, from_zarr
from .lazyesn import LazyESN
