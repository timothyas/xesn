# Figure out if we have cupy/GPU access
try:
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()
    _use_cupy = True

except ImportError:
    _use_cupy = False

except xp.cuda.runtime.CUDARuntimeError:
    _use_cupy = False

from .driver import Driver
from .esn import ESN
from .lazyesn import LazyESN
from .io import from_zarr

from .matrix import RandomMatrix, SparseRandomMatrix
from .cost import CostFunction
from .optim import optimize
from .psd import psd
from .xdata import XData
