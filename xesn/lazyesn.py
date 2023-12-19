import json
from functools import reduce
import xarray as xr
from dask.array import map_blocks, stack
from dask.array.overlap import overlap

from . import _use_cupy
if _use_cupy:
    import cupy as xp
    from cupy.linalg import solve

else:
    import numpy as xp
    from scipy.linalg import solve

from .esn import ESN, _train_1d, _update

class LazyESN(ESN):
    """A distributed/parallelized ESN network based on the multi-dimensional generalization of
    the algorithm introduced by :cite:t:`pathak_model-free_2018`, as used in
    :cite:t:`smith_temporal_2023`.

    Assumptions:
        1. For all data provided to ESN methods, time axis is last and it is named "time"
        2. Non-global axes, i.e., axes which is chunked up or made up of patches, are first
        3. Can handle multi-dimensional data, but only 2D chunking

    Note:
        The difference between what is expected for ``overlap`` and ``boundary``
        and what is supplied to dask's overlap function is that here the
        dimensions are labelled. This means we expect something like
        ``{"x": 1, "y":1, "time": 0}`` rather than ``{0: 1, 1:1, 2:0}``, for
        ``overlap`` and similar for ``boundary``.

    Args:
        esn_chunks (dict): mapping the input data dimension names to its chunksize
        overlap (dict): mapping the input data dimension names to the number of neighboring points in each dimension to include in each local input vector
        boundary (dict, str, or float): indicate how to handle the domain boundaries during overlap. Available options are ``"periodic"``, ``"reflect"``, or ``value`` where ``value`` is a value to be filled into the domain, which could be ``np.nan`` for hard boundaries.
        n_reservoir (int): size of the reservoir or hidden state for each local network
        leak_rate (float): fraction of current hidden state to use during timestepping, ``(1-leak_rate) r(n-1)`` is propagated forward
        tikhonov_parameter (float): regularization parameter to prevent overfitting
        persist (bool, optional): if True, bring the data into memory using all available computational resources. This is called after calling overlap in :meth:`train` and :meth:`predict`, as well as on the readout weights :attr:`Wout` after training and on the prediction result after computations are complete in :meth:`predict`
        input_kwargs (dict, optional): the options to specify :attr:`Win`, use boolean option ``"is_sparse"`` to determine if :class:`RandomMatrix` or :class:`SparseRandomMatrix` is used, then all other options are passed to either of those classes, see their description for available options noting that ``n_rows`` and ``n_cols`` are not necessary.
        adjacency_kwargs (dict, optional): the options to specify :attr:`W`, use boolean option ``"is_sparse"`` to determine if :class:`RandomMatrix` or :class:`SparseRandomMatrix` is used, then all other options are passed to either of those classes, see their description for available options noting that ``n_rows`` and ``n_cols`` are not necessary.
        bias_kwargs (dict, optional): the options to specifying :attr:`bias_vector` generation. Only ``"distribution"``, ``"factor"``, and ``"random_seed"`` options are allowed.
    """

    __slots__ = (
        "esn_chunks", "overlap", "persist", "boundary"
    )

    @property
    def output_chunks(self):
        return self.esn_chunks

    @property
    def input_chunks(self):
        return {k: self.output_chunks[k]+2*self.overlap[k] for k in self.output_chunks.keys()}

    @property
    def ndim_state(self):
        """Num. of non-time axes"""
        return len(self.overlap)-1

    @property
    def _r_chunks(self):
        """The number of dimensions needs to be the same as the original multi-dimensional data"""
        c = tuple(1 for _ in range(self.ndim_state-1))
        c += (self.n_reservoir,)
        return c

    @property
    def _Wout_chunks(self):
        chunks = (self.n_output, self.n_reservoir)
        for _ in range(self.ndim_state - 2):
            chunks += (1,)
        return chunks


    def __init__(self,
            esn_chunks,
            overlap,
            boundary,
            n_reservoir,
            leak_rate,
            tikhonov_parameter,
            persist=False,
            input_kwargs=None,
            adjacency_kwargs=None,
            bias_kwargs=None):

        # Figure out input/output from data chunk sizes
        self.overlap        = overlap
        self.boundary       = boundary
        self.persist        = persist
        self.esn_chunks     = esn_chunks

        # "time" doesn't have to be in overlap or esn_chunks,
        # since it's trivially a single chunk/no overlap
        # make that assumption here
        if "time" not in overlap:
            self.overlap["time"] = 0

        if "time" not in esn_chunks:
            self.esn_chunks["time"] = -1

        # We can't have -1's in the spatial esn_chunks,
        # because we're taking products to compute sizes
        if any(x < 0 for k, x in self.esn_chunks.items() if k != "time"):
            raise ValueError("LazyESN.__init__: Cannot have negative numbers or Nones in non-temporal axis locations of esn_chunks. Provide the actual value please.")

        n_output = _prod([x for k, x in self.output_chunks.items() if k!="time"])
        n_input = _prod([x for k, x in self.input_chunks.items() if k!="time"])

        super().__init__(
                n_input=n_input,
                n_output=n_output,
                n_reservoir=n_reservoir,
                leak_rate=leak_rate,
                tikhonov_parameter=tikhonov_parameter,
                input_kwargs=input_kwargs,
                adjacency_kwargs=adjacency_kwargs,
                bias_kwargs=bias_kwargs)

        try:
            assert len([axis for axis, depth in self.overlap.items() if depth > 0]) <= 2

        except AssertionError:
            raise NotImplementedError(f"LazyESN.__init__: cannot overlap more than 2 axes")


    def __str__(self):

        boundary = "\n"+self._dictstr(self.boundary) if isinstance(self.boundary, dict) else str(self.boundary)
        rstr = 'LazyESN\n'+\
                f'    input_chunks:\n{self._dictstr(self.input_chunks)}'+\
                 '---\n'+\
                f'    output_chunks:\n{self._dictstr(self.output_chunks)}'+\
                 '---\n'+\
                f'    overlap:\n{self._dictstr(self.overlap)}'+\
                 '---\n'+\
                f'    {"boundary:":<24s}{boundary}\n'+\
                 '---\n'+\
                 super().__str__().replace("ESN\n","")
        return rstr


    def train(self, y, n_spinup=0, batch_size=None):
        """Learn the readout matrix weights through ridge regression.

        Args:
            y (xarray.DataArray): target or label data for training, if different from the input data
            n_spinup (int, optional): number of spinup steps for the ESN, not included in training
            batch_size (int, optional): used to perform training in batches,
                but note that all time data are still loaded into memory regardless of this parameter

        Sets Attributes:
            Wout (array_like): the readout matrix, mapping from reservoir to output space
        """

        self._time_check(y, "LazyESN.train", "y")

        doverlap = self._dask_overlap(y.dims)
        dboundary= self._dask_boundary(y.dims)
        target_data = y.chunk(self.output_chunks).data
        halo_data = overlap(target_data, depth=doverlap, boundary=dboundary, allow_rechunk=False)
        halo_data = halo_data.persist() if self.persist else halo_data

        self.Wout = map_blocks(
                _train_nd,
                halo_data,
                overlap=doverlap,
                n_spinup=n_spinup,
                batch_size=batch_size,
                W=self.W,
                Win=self.Win,
                bias_vector=self.bias_vector,
                leak_rate=self.leak_rate,
                tikhonov_parameter=self.tikhonov_parameter,
                drop_axis=-1,
                chunks=self._Wout_chunks,
                enforce_ndim=True,
                dtype=xp.float64,
                meta=xp.array((), dtype=xp.float64),
        )

        self.Wout = self.Wout.persist() if self.persist else self.Wout


    def predict(self, y, n_steps, n_spinup):

        self._time_check(y, "LazyESN.predict", "y")

        assert y.shape[-1] >= n_spinup+1

        # Get overlapped data
        doverlap = self._dask_overlap(y.dims)
        dboundary = self._dask_boundary(y.dims)
        target_data = y[..., :n_spinup+1].chunk(self.output_chunks).data
        halo_data = overlap(target_data, depth=doverlap, boundary=dboundary)
        halo_data = halo_data.persist() if self.persist else halo_data

        ukw = { "W"             : self.W,
                "Win"           : self.Win,
                "bias_vector"   : self.bias_vector,
                "leak_rate"     : self.leak_rate,
                }

        dkw = { "enforce_ndim"  : True,
                "dtype"         : xp.float64,
                "meta"          : xp.ndarray((), xp.float64),
                }

        # Spinup
        r0 = map_blocks(
                _spinup,
                halo_data,
                n_spinup=n_spinup,
                chunks=self._r_chunks,
                drop_axis=-1, # drop time axis
                **ukw, **dkw)

        # Necessary for 1D output, since Wout is at least 2D
        drop_axis = None if self.ndim_state > 1 else 0

        # Setup and loop
        u0 = halo_data[..., n_spinup]
        prediction = [target_data[..., n_spinup]]
        chunksize = target_data[..., 0].chunksize
        for n in range(n_steps):

            r0 = map_blocks(_update_nd, r0, u0, chunks=self._r_chunks, **ukw, **dkw)
            v  = map_blocks(_readout, self.Wout, r0, chunks=chunksize, drop_axis=drop_axis, **dkw)

            u0 = overlap(v, depth=doverlap, boundary=dboundary)
            prediction.append(v)

        # Stack, rechunk, persist, return
        prediction = stack(prediction, axis=-1)
        prediction = prediction.rechunk({-1:-1})
        prediction = prediction.persist() if self.persist else prediction

        fdims, fcoords = self._get_fcoords(y.dims, y.coords, n_steps, n_spinup)
        xpred = xr.DataArray(
                prediction,
                coords=fcoords,
                dims=fdims)
        return xpred


    def _dask_overlap(self, dims):
        """To use dask.overlap, we need a dictionary referencing axis indices, not
        named dimensions as with xarray. Create that index based dict here"""
        return {dims.index(d): self.overlap[d] for d in self.overlap.keys()}


    def _dask_boundary(self, dims):
        if isinstance(self.boundary, dict):
            return {dims.index(d): self.boundary[d] for d in self.boundary.keys()}
        else:
            return self.boundary


def _train_nd(halo_data,
        overlap=None,
        n_spinup=0,
        batch_size=None,
        W=None,
        Win=None,
        bias_vector=None,
        leak_rate=None,
        tikhonov_parameter=None,
        block_info=None,
        ):

    # Deal with overlap related masking
    # Note: how to put this stuff in _prepare_1d_inputs, even though we only need and use the inner mask here?
    u = _flatten_space( halo_data )
    inner_mask = _get_inner_mask(halo_data.shape, overlap)
    y = u[inner_mask.flatten(), :]

    # Get sizes before masking
    n_output, _ = y.shape
    n_reservoir, _ = Win.shape

    # Deal with domain boundaries (i.e., NaNs outside bounds of non-periodic global domain)
    bdy_mask = xp.isnan(u[:,0])
    u   = u[~bdy_mask, :]
    Win = Win[:, ~bdy_mask]

    # Deal with internal boundaries (i.e., NaNs inside domain, like continental boundaries for ocean)
    ibdy_mask = xp.isnan(y[:,0])
    y   = y[~ibdy_mask, :]

    # Make and fill container
    Wout = xp.full( (n_output, n_reservoir), xp.nan, dtype=xp.float64 )
    Wout[~ibdy_mask, :] = _train_1d(u, y, n_spinup, batch_size, W, Win, bias_vector, leak_rate, tikhonov_parameter)
    return Wout.reshape(block_info[None]["chunk-shape"])


def _spinup(halo_data,
        n_spinup=0,
        W=None,
        Win=None,
        bias_vector=None,
        leak_rate=None,
        block_info=None,
        ):

    u, Win = _prepare_1d_inputs(halo_data, Win, has_time=True)
    uT = u.T
    r = xp.zeros(shape=(W.shape[0],))
    for n in range(n_spinup):
        r = _update(r, uT[n], W, Win, bias_vector, leak_rate)

    return r.reshape(block_info[None]["chunk-shape"])


def _update_nd(r, halo_data, W=None, Win=None, bias_vector=None, leak_rate=None, block_info=None):
    u, Win = _prepare_1d_inputs(halo_data, Win, has_time=False)
    rp1 = _update(r.squeeze(), u.squeeze(), W, Win, bias_vector, leak_rate)
    return rp1.reshape(r.shape)


def _readout(Wout, r, block_info=None):
    v = Wout.squeeze() @ r.squeeze()
    return v.reshape(block_info[None]["chunk-shape"])


def _get_inner_mask(shape, overlap):
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


def _prepare_1d_inputs(halo_data, Win, has_time):

    # Deal with domain boundaries
    halo_data = halo_data[...,None] if not has_time else halo_data
    u = _flatten_space( halo_data )
    bdy_mask = xp.isnan(u[:, 0])
    u   = u[~bdy_mask, :].squeeze()
    Win = Win[:, ~bdy_mask]
    return u, Win


def _prod(array):
    return reduce( lambda x,y: x*y, array )
