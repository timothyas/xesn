from functools import reduce
from dask.array import map_blocks
from dask.array.overlap import overlap

from . import _use_cupy
if _use_cupy:
    import cupy as xp
    from cupy.linalg import solve

else:
    import numpy as xp
    from scipy.linalg import solve

from .esn import ESN, _update

class LazyESN(ESN):

    @property
    def halo_axes(self):
        return tuple(axis for axis, depth in self.overlap.items() if depth > 0)

    def __init__(self,
            data_chunks,
            n_reservoir,
            input_factor,
            adjacency_factor,
            connectedness,
            bias,
            leak_rate,
            tikhonov_parameter,
            overlap,
            persist,
            boundary=xp.nan,
            input_kwargs=None,
            adjacency_kwargs=None,
            random_seed=None):

        # Figure out input/output from data chunk sizes
        self.output_chunks = data_chunks
        self.input_chunks = tuple(n+2*o for n,o in zip(data_chunks, overlap.values()))
        n_output = _prod(self.output_chunks[:-1])
        n_input = _prod(self.input_chunks[:-1])
        super().__init__(
                n_input=n_input,
                n_output=n_output,
                n_reservoir=n_reservoir,
                input_factor=input_factor,
                adjacency_factor=adjacency_factor,
                connectedness=connectedness,
                bias=bias,
                leak_rate=leak_rate,
                tikhonov_parameter=tikhonov_parameter,
                input_kwargs=input_kwargs,
                adjacency_kwargs=adjacency_kwargs,
                random_seed=random_seed)

        self.overlap    = overlap
        self.boundary   = boundary
        self.persist    = persist


    def train(self, u, n_spinup=0, batch_size=None):
        """
        Args:
            u (dask.array): n_state1, n_state2, ..., n_time
        """

        halo_data = overlap(u, depth=self.overlap, boundary=self.boundary)
        halo_data = halo_data.persist() if self.persist else halo_data

        self.Wout = map_blocks(
                _train,
                halo_data,
                overlap=self.overlap,
                n_spinup=n_spinup,
                batch_size=batch_size,
                W=self.W,
                Win=self.Win,
                bias_vector=self.bias_vector,
                leak_rate=self.leak_rate,
                tikhonov_parameter=self.tikhonov_parameter,
                chunks=(self.n_output,self.n_reservoir),
                enforce_ndim=True,
                dtype=xp.float64,
        )

        self.Wout = self.Wout.persist() if self.persist else self.Wout


def _train(halo_data,
        overlap=None,
        n_spinup=0,
        batch_size=None,
        W=None,
        Win=None,
        bias_vector=None,
        leak_rate=None,
        tikhonov_parameter=None,
        ):
    """
    TODO:
        - Deal with input mask different from halo region, halo with NaNs
        - Deal with output mask for nontrivial boundaries

    Args:
        halo_data (dask.array): n_state1, n_state2, ..., n_time
    """



    # Deal with masking
    u = _flatten_space( halo_data )
    mask_nd = _get_active_mask(halo_data.shape, overlap)
    mask_1d = mask_nd.flatten()
    y = u[mask_1d, :]

    # Important Integers
    n_reservoir, n_input = Win.shape
    n_output, n_time = y.shape

    batch_size = n_time if batch_size is None else batch_size
    n_batches = int(xp.ceil( (n_time - n_spinup) / batch_size ))

    # Make containers
    uT = u.T
    rT = xp.zeros(
            shape=(batch_size+1, n_reservoir))
    ybar = xp.zeros(
            shape=(n_output, n_reservoir))
    rbar = xp.zeros(
            shape=(n_reservoir, n_reservoir))

    kw = {
            "W"             : W,
            "Win"           : Win,
            "bias_vector"   : bias_vector,
            "leak_rate"     : leak_rate}

    # Spinup
    for n in range(n_spinup):
        rT[0] = _update(rT[0], uT[n], **kw)

    # Accumulate matrices
    for i in range( n_batches ):
        i0 = i*batch_size + n_spinup
        i1 = min((i+1)*batch_size + n_spinup, n_time)

        for n, n_in in enumerate(range(i0, i1)):
            rT[n+1] = _update(rT[n], uT[n_in], **kw)

        ybar += y[:, i0:i1] @ rT[:n+1, :]
        rbar += rT[:n+1, :].T @ rT[:n+1, :]

        # Start over for next batch
        rT[0] = rT[n+1].copy()

    # Linear solve
    rbar += tikhonov_parameter * xp.eye(n_reservoir)

    kw = {} if _use_cupy else {"assume_a": "sym"}
    Wout = solve(rbar.T, ybar.T, **kw)
    return Wout.T


def _get_active_mask(shape, overlap):
    """operate on each chunk/block to show False if in overlap, True if not"""

    inner_mask = [
        _inner_1d(n=shape[axis], depth=depth) for axis, depth in overlap.items() if depth>0
    ]

    inner_mask = xp.outer(*inner_mask) if len(inner_mask)>1 else inner_mask[0]

    new_shape = shape[:-1] + (1,)
    inner_mask = xp.broadcast_to(inner_mask.T, new_shape[::-1]).T
    return inner_mask


def _inner_1d(n, depth):
    """create a single axis 1D mask, True denoting inner region"""
    single = xp.full(n, True, dtype=bool)
    single[:depth]  = False
    single[-depth:] = False
    return single


def _flatten_space(arr):
    shape = (_prod(arr.shape[:-1]), arr.shape[-1])
    return arr.reshape(shape)


def _prod(array):
    return reduce( lambda x,y: x*y, array )
