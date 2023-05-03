import warnings
import inspect
import xarray as xr

from . import _use_cupy
if _use_cupy:
    import cupy as xp
    from cupy.linalg import solve

else:
    import numpy as xp
    from scipy.linalg import solve

from .matrix import RandomMatrix, SparseRandomMatrix

class ESN():
    W                   = None
    Win                 = None
    Wout                = None
    input_kwargs        = None
    adjacency_kwargs    = None

    @property
    def density(self):
        return self.connectedness / self.n_reservoir

    @property
    def sparsity(self):
        return 1. - self.density

    @property
    def input_factor(self):
        return self.input_kwargs["factor"]

    @property
    def adjacency_factor(self):
        return self.adjacency_kwargs["factor"]

    def __init__(self,
            n_input,
            n_output,
            n_reservoir,
            input_factor,
            adjacency_factor,
            connectedness,
            bias,
            leak_rate,
            tikhonov_parameter,
            input_kwargs=None,
            adjacency_kwargs=None,
            random_seed=None):

        # Required inputs
        self.n_input            = n_input
        self.n_output           = n_output
        self.n_reservoir        = n_reservoir
        self.connectedness      = connectedness
        self.bias               = bias
        self.leak_rate          = leak_rate
        self.tikhonov_parameter = tikhonov_parameter
        self.random_seed        = random_seed
        self.random_state       = xp.random.RandomState(self.random_seed)

        # Handle input and adjacency matrices
        self.input_kwargs = {
            "factor"        : input_factor,
            "distribution"  : "uniform",
            "normalization" : "multiply",
            "is_sparse"     : False,
            }
        if input_kwargs is not None:
            self.input_kwargs.update(input_kwargs)

        self.adjacency_kwargs = {
            "factor"        : adjacency_factor,
            "density"       : self.density,
            "distribution"  : "uniform",
            "normalization" : "eig",
            "is_sparse"     : True,
            "format"        : "csr",
            }
        if adjacency_kwargs is not None:
            self.adjacency_kwargs.update(adjacency_kwargs)
            if not self.adjacency_kwargs["is_sparse"]:
                raise NotImplementedError

        # Check inputs
        try:
            assert self.bias >= 0.0
        except AssertionError:
            raise ValueError(f"ESN.__init__: bias must be non-negative, got {self.bias}")

        try:
            assert self.connectedness < self.n_reservoir
        except AssertionError:
            raise ValueError(f"ESN.__init__: connectedness must be < n_reservoir, got {self.connectedness}")

        if self.adjacency_kwargs["is_sparse"] and self.sparsity < 0.8:
            warnings.warn(f"ESN.__init__: sparsity is below 80% but adjacency_kwargs['is_sparse'] = {self.adjacency_kwargs['is_sparse']}. Performance could suffer from dense matrix operations with scipy.sparse.", RuntimeWarning)

        if _use_cupy and adjacency_kwargs["normalization"] == "eig":
            raise ValueError(f"ESN.__init__: with cupy, cannot use eigenvalues to normalize matrices, use 'svd'")


    def __str__(self):
        rstr = 'ESN\n'+\
                f'    {"n_input:":<24s}{self.n_input}\n'+\
                f'    {"n_output:":<24s}{self.n_output}\n'+\
                f'    {"n_reservoir:":<24s}{self.n_reservoir}\n'+\
                 '--- \n'+\
                f'    {"connectedness:":<24s}{self.connectedness}\n'+\
                f'    {"bias:":<24s}{self.bias}\n'+\
                f'    {"leak_rate:":<24s}{self.leak_rate}\n'+\
                f'    {"tikhonov_parameter:":<24s}{self.tikhonov_parameter}\n'+\
                 '--- \n'+\
                f'    {"random_seed:":<24s}{self.random_seed}\n'+\
                f'    {"random_state:":<24s}{self.random_state}\n'+\
                 '--- \n'+\
                f'    Input Matrix:\n'
        for key, val in self.input_kwargs.items():
            rstr += f'        {key:<20s}{val}\n'

        rstr += \
                 '--- \n'+\
                f'    Adjacency Matrix:\n'
        for key, val in self.adjacency_kwargs.items():
            rstr += f'        {key:<20s}{val}\n'

        return rstr


    def __repr__(self):
        return self.__str__()


    def build(self):
        """Generate the random adjacency and input weight matrices
        with sparsity determined by :attr:`sparsity` attribute,
        scaled by :attr:`spectral_radius` and :attr:`sigma` parameters, respectively.

        Returns and Sets Attributes:
            A (array_like): (:attr:`n_reservoir`, :attr:`n_reservoir`),
                reservoir adjacency matrix
            Win (array_like): (:attr:`n_reservoir`, :attr:`n_input`),
                reservoir input weight matrix
        """

        is_sparse = self.adjacency_kwargs.pop("is_sparse", True)
        Matrix = SparseRandomMatrix if is_sparse else RandomMatrix
        WMaker = Matrix(
                n_rows=self.n_reservoir,
                n_cols=self.n_reservoir,
                random_state=self.random_state,
                **self.adjacency_kwargs)
        self.W = WMaker()


        is_sparse = self.input_kwargs.pop("is_sparse", False)
        Matrix = SparseRandomMatrix if is_sparse else RandomMatrix
        WinMaker = Matrix(
                n_rows=self.n_reservoir,
                n_cols=self.n_input,
                random_state=self.random_state,
                **self.input_kwargs)
        self.Win = WinMaker()

        self.bias_vector = self.random_state.uniform(low=-self.bias,
                                                     high=self.bias,
                                                     size=(self.n_reservoir,))


    def train(self, u, y=None, n_spinup=0, batch_size=None):

        # Check if training labels are different from input data
        y = u if y is None else y

        self.Wout = _train_1d(
                u, y, n_spinup, batch_size,
                self.W, self.Win, self.bias_vector, self.leak_rate,
                self.tikhonov_parameter)


    def predict(self, u, n_steps, n_spinup):

        uT = u.T
        _, n_time = u.shape
        assert n_time >= n_spinup

        # Make containers
        r = xp.zeros(shape=(self.n_reservoir,))
        yT = xp.zeros(
                shape=(n_steps+1, self.n_output))
        kw = {
                "W"             : self.W,
                "Win"           : self.Win,
                "bias_vector"   : self.bias_vector,
                "leak_rate"     : self.leak_rate}

        # Spinup
        for n in range(n_spinup):
            r = _update(r, uT[n], **kw)

        # Prediction
        yT[0] = uT[n_spinup]
        for n in range(1, n_steps+1):
            r = _update(r, yT[n-1], **kw)
            yT[n] = self.Wout @ r

        return yT.T


    def to_xds(self):
        """Return object as :obj:`xarray.Dataset`

        Note:
            For now, not storing :attr:`W` or :attr:`Win`. Instead, store :attr:`random_seed`.
        """
        import xarray as xr

        ds = xr.Dataset()
        ir = xp.arange(self.Wout.squeeze().shape[-1])
        ir = ir.get() if _use_cupy else ir
        ds['ir'] = xr.DataArray(ir, coords={'ir': ir}, dims=('ir',), attrs={'description': 'logical index for reservoir coordinate'})

        iy = xp.arange(self.Wout.squeeze().shape[0])
        iy = iy.get() if _use_cupy else iy
        ds['iy'] = xr.DataArray(iy, coords={'iy': iy}, dims=('iy',), attrs={'description': 'logical index for flattened output axis'})

        # the main stuff
        dims = ("iy", "ir")
        Wout = self.Wout.get() if _use_cupy else self.Wout
        ds["Wout"] = xr.DataArray(Wout, coords={k: ds[k] for k in dims}, dims=dims)

        # everything else
        kw, *_ = inspect.getfullargspec(self.__init__)
        kw.remove("self")

        for key in kw:
            val = getattr(self, key)
            if isinstance(val, bool) or val is None:
                ds.attrs[key] = str(val)
            else:
                ds.attrs[key] = val

        return ds


def _update(r, u, W, Win, bias_vector, leak_rate):
    p = W @ r + Win @ u + bias_vector
    return leak_rate * xp.tanh(p) + (1-leak_rate) * r


def _train_1d(
        u,
        y,
        n_spinup,
        batch_size,
        W,
        Win,
        bias_vector,
        leak_rate,
        tikhonov_parameter,
        ):

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
