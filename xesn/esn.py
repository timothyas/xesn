import warnings
import inspect
from decimal import Decimal
import xarray as xr
import dask.array as darray
import numpy as np


from . import _use_cupy
if _use_cupy:
    import cupy as xp
    from cupy.linalg import solve
    import cupy_xarray

else:
    import numpy as xp
    from scipy.linalg import solve

from .matrix import RandomMatrix, SparseRandomMatrix

class ESN():
    """A classic ESN architecture, as introduced by :cite:t:`jaeger_echo_2001`,
    with no distribution or parallelism.
    It is assumed that all data used with this architecture can fit into memory.

    Assumptions:
        1. For all data provided to ESN methods, time axis is last and it is named "time"

    Args:
        n_input (int): size of the input vector to the ESN in state space
        n_output (int): size of the ESN output vector in state space
        n_reservoir (int): size of the reservoir or hidden state
        leak_rate (float): fraction of current hidden state to use during timestepping, ``(1-leak_rate) r(n-1)`` is propagated forward
        tikhonov_parameter (float): regularization parameter to prevent overfitting
        input_kwargs (dict, optional): the options to specify :attr:`Win`, use boolean option ``"is_sparse"`` to determine if :class:`RandomMatrix` or :class:`SparseRandomMatrix` is used, then all other options are passed to either of those classes, see their description for available options noting that ``n_rows`` and ``n_cols`` are not necessary.
        adjacency_kwargs (dict, optional): the options to specify :attr:`W`, use boolean option ``"is_sparse"`` to determine if :class:`RandomMatrix` or :class:`SparseRandomMatrix` is used, then all other options are passed to either of those classes, see their description for available options noting that ``n_rows`` and ``n_cols`` are not necessary.
        bias_kwargs (dict, optional): the options to specifying :attr:`bias_vector` generation. Only ``"distribution"``, ``"factor"``, and ``"random_seed"`` options are allowed.
    """

    __slots__ = (
        "W", "Win", "Wout",
        "n_input", "n_output", "n_reservoir",
        "leak_rate", "tikhonov_parameter", "bias_vector",
        "input_kwargs", "adjacency_kwargs", "bias_kwargs",
    )

    @property
    def input_factor(self):
        return self.input_kwargs["factor"]

    @property
    def adjacency_factor(self):
        return self.adjacency_kwargs["factor"]

    @property
    def bias_factor(self):
        return self.bias_kwargs["factor"]

    def __init__(self,
            n_input,
            n_output,
            n_reservoir,
            leak_rate,
            tikhonov_parameter,
            input_kwargs=None,
            adjacency_kwargs=None,
            bias_kwargs=None):

        # Required inputs
        self.n_input            = n_input
        self.n_output           = n_output
        self.n_reservoir        = n_reservoir
        self.leak_rate          = leak_rate
        self.tikhonov_parameter = tikhonov_parameter

        # Handle input matrix options
        default_input_kwargs = {
                "factor"        : 1.0,
                "distribution"  : "uniform",
                "normalization" : "multiply",
                "is_sparse"     : False,
                "random_seed"   : None,
                }

        default_adjacency_kwargs = {
                "factor"        : 1.0,
                "distribution"  : "uniform",
                "normalization" : "eig",
                "is_sparse"     : True,
                "connectedness" : 5,
                "format"        : "csr",
                "random_seed"   : None,
                }

        default_bias_kwargs = {
                "factor"        : 1.0,
                "distribution"  : "uniform",
                "random_seed"   : None,
                }

        self.input_kwargs = self._set_matrix_options(input_kwargs, default_input_kwargs, "input_kwargs")
        self.adjacency_kwargs = self._set_matrix_options(adjacency_kwargs, default_adjacency_kwargs, "adjacency_kwargs")
        self.bias_kwargs = self._set_matrix_options(bias_kwargs, default_bias_kwargs, "bias_kwargs")

        # Check inputs
        try:
            assert self.bias_factor >= 0.0
        except AssertionError:
            raise ValueError(f"ESN.__init__: bias_factor must be non-negative, got {self.bias_factor}")

        if _use_cupy:
            normalization = adjacency_kwargs.get("normalization", "multiply")
            if normalization == "eig":
                raise ValueError(f"ESN.__init__: with cupy, cannot use eigenvalues to normalize matrices, use 'svd'")


    @staticmethod
    def _dictstr(mydict):
        lefttab = "        "
        dstr = ""
        for key, val in mydict.items():
            dstr += f"{lefttab}{key:<20s}{val}\n"
        return dstr


    def __str__(self):
        rstr = 'ESN\n'+\
                f'    {"n_input:":<24s}{self.n_input}\n'+\
                f'    {"n_output:":<24s}{self.n_output}\n'+\
                f'    {"n_reservoir:":<24s}{self.n_reservoir}\n'+\
                 '---\n'+\
                f'    {"leak_rate:":<24s}{self.leak_rate}\n'+\
                f'    {"tikhonov_parameter:":<24s}{self.tikhonov_parameter}\n'+\
                 '---\n'+\
                f'    Input Matrix:\n{self._dictstr(self.input_kwargs)}'+\
                 '---\n'+\
                f'    Adjacency Matrix:\n{self._dictstr(self.adjacency_kwargs)}'+\
                 '---\n'+\
                f'    Bias Vector:\n{self._dictstr(self.bias_kwargs)}'
        return rstr


    def __repr__(self):
        return self.__str__()


    def build(self):
        """Generate the random adjacency and input weight matrices
        with sparsity determined by :attr:`sparsity` attribute,
        scaled by :attr:`spectral_radius` and :attr:`sigma` parameters, respectively.

        Sets Attributes:
            A (array_like): (:attr:`n_reservoir`, :attr:`n_reservoir`),
                reservoir adjacency matrix
            Win (array_like): (:attr:`n_reservoir`, :attr:`n_input`),
                reservoir input weight matrix
        """

        # Note: this copy is necessary because we want to remove "is_sparse"
        # before passing to either Matrix class, but we want to keep "is_sparse"
        # in case the user stores the ESN to zarr
        kw = self.adjacency_kwargs.copy()
        is_sparse = kw.pop("is_sparse", False)
        Matrix = SparseRandomMatrix if is_sparse else RandomMatrix
        WMaker = Matrix(
                n_rows=self.n_reservoir,
                n_cols=self.n_reservoir,
                **kw)
        self.W = WMaker()
        if is_sparse and WMaker.density > 0.2:
            warnings.warn(f"ESN.__init__: adjacency matrix density is >20% but adjacency_kwargs['is_sparse'] = True. Performance could suffer from dense matrix operations with scipy.sparse.", RuntimeWarning)


        kw = self.input_kwargs.copy()
        is_sparse = kw.pop("is_sparse", False)
        Matrix = SparseRandomMatrix if is_sparse else RandomMatrix
        WinMaker = Matrix(
                n_rows=self.n_reservoir,
                n_cols=self.n_input,
                **kw)
        self.Win = WinMaker()

        BiasMaker = RandomMatrix(
                n_rows=1,
                n_cols=self.n_reservoir,
                **self.bias_kwargs)
        self.bias_vector = BiasMaker().squeeze()


    def train(self, u, y=None, n_spinup=0, batch_size=None):
        """Learn the readout matrix weights through ridge regression.

        Args:
            u (xarray.DataArray): input data driving the ESN, with "time" as the last dimension
            y (xarray.DataArray, optional): target or label data for training, if different from the input data
            n_spinup (int, optional): number of spinup steps for the ESN, not included in training
            batch_size (int, optional): used to perform training in batches,
                but note that all time data are still loaded into memory regardless of this parameter

        Sets Attributes:
            Wout (array_like): (:attr:`n_ouput`, :attr:`n_reservoir`)
                the readout matrix, mapping from reservoir to output space
        """

        # Check if training labels are different from input data
        y = u if y is None else y
        self._time_check(u, "ESN.train", "u")
        self._time_check(y, "ESN.train", "y")

        n_time = y.shape[1]
        assert n_time >= n_spinup

        # load the data
        u = u.load()
        y = y.load()

        self.Wout = _train_1d(
                u.data, y.data, n_spinup, batch_size,
                self.W, self.Win, self.bias_vector, self.leak_rate,
                self.tikhonov_parameter)


    def predict(self, y, n_steps, n_spinup):
        """Use the ESN to make a prediction

        Note:
            This creates a new ``ftime`` dimension, indicating the time since prediction initial conditions (forecast time). The ``ftime`` vector is created by subtraction: ``y["time"].values - y["time"].values[n_spinup]``. If ``y["time"]`` is filled with floats, it is recommended to add the attribute: ``y["time"].attrs["delta_t"]`` indicating the finest increment to round ftime to. Otherwise, floating point arithmetic will make this vector have crazy values.

        Args:
            y (xarray.DataArray): the input data driving the reservoir during spinup, must have "time" as the last dimension,
                and it needs to have at least ``n_spinup`` entries in time
            n_steps (int): number of prediction steps to take
            n_spinup (int): number of spinup steps before making the prediction

        Returns:
            xpred (xarray.DataArray): the prediction, with no spinup data and length ``n_steps+1`` along
                the newly created ``ftime`` dimension, created by differencing each timestamp and time at
                prediction initial conditions
        """

        self._time_check(y, "ESN.predict", "y")

        # load the data
        y = y.load()

        yT = y.data.T
        _, n_time = y.shape
        assert n_time >= n_spinup

        # Make containers
        r = xp.zeros(shape=(self.n_reservoir,))
        vT = xp.zeros(
                shape=(n_steps+1, self.n_output))
        kw = {
                "W"             : self.W,
                "Win"           : self.Win,
                "bias_vector"   : self.bias_vector,
                "leak_rate"     : self.leak_rate}

        # Spinup
        for n in range(n_spinup):
            r = _update(r, yT[n], **kw)

        # Prediction
        vT[0] = yT[n_spinup]
        for n in range(1, n_steps+1):
            r = _update(r, vT[n-1], **kw)
            vT[n] = self.Wout @ r

        fdims, fcoords = self._get_fcoords(y.dims, y.coords, n_steps, n_spinup)
        xpred = xr.DataArray(
                vT.T,
                coords=fcoords,
                dims=fdims)

        return xpred


    def test(self, y, n_steps, n_spinup):
        """Make a prediction to be compared to a truth. The only difference
        with :meth:`predict` is that this returns a dataset with both the prediction and truth.

        Note:
            This creates a new ``ftime`` dimension, indicating the time since prediction initial conditions (forecast time). The ``ftime`` vector is created by subtraction: ``y["time"].values - y["time"].values[n_spinup]``. If ``y["time"]`` is filled with floats, it is recommended to add the attribute: ``y["time"].attrs["delta_t"]`` indicating the finest increment to round ftime to. Otherwise, floating point arithmetic will make this vector have crazy values.

        Args:
            y (xarray.DataArray): the input data driving the reservoir during spinup, must have "time" as the last dimension,
                and it needs to have at least ``n_spinup`` entries in time
            n_steps (int): number of prediction steps to take
            n_spinup (int): number of spinup steps before making the prediction

        Returns:
            xds (xarray.Dataset): with fields "prediction" and "truth", see :meth:`predict`
        """

        # make prediction
        xds = xr.Dataset()
        xds["prediction"] = self.predict(y, n_steps, n_spinup)
        xds["truth"] = xr.DataArray(
                y.sel(time=xds.prediction.time).data,
                coords=xds.prediction.coords,
                dims=xds.prediction.dims)
        xds.attrs.update(self._get_attrs())
        xds.attrs["esn_type"] = self.__class__.__name__
        xds.attrs["description"] = "Contains a test prediction and matching truth trajectory"
        return xds


    def to_xds(self):
        """Return object as :obj:`xarray.Dataset`

        Note:
            This does not store :attr:`W` or :attr:`Win`. Instead, store the random seed for each within kwargs.

        Returns:
            xds (xarray.Dataset): with field "Wout" containing the readout matrix, and attributes
                that can recreate the ESN
        """

        if self.Wout is None:
            raise Exception("ESN.to_xds: Wout has not been computed yet, so it's not worth storing this model")

        ds = xr.Dataset()
        ir = np.arange(self.Wout.squeeze().shape[-1])
        ds['ir'] = xr.DataArray(ir, coords={'ir': ir}, dims=('ir',), attrs={'description': 'logical index for reservoir coordinate'})

        iy = np.arange(self.Wout.squeeze().shape[0])
        ds['iy'] = xr.DataArray(iy, coords={'iy': iy}, dims=('iy',), attrs={'description': 'logical index for flattened output axis'})

        # the main stuff
        dims = ("iy", "ir")
        ds["Wout"] = xr.DataArray(self.Wout.squeeze(), coords={k: ds[k] for k in dims}, dims=dims)



        # everything else
        ds.attrs.update(self._get_attrs())
        return ds


    @staticmethod
    def _set_matrix_options(user, default, name):
        """Handle variety of possible user provided matrix options"""

        allowed = {
                "input_kwargs": [
                    "factor",
                    "distribution",
                    "normalization",
                    "is_sparse",
                    "density",
                    "sparsity",
                    "connectedness",
                    "format",
                    "random_seed",
                    ],
                "adjacency_kwargs": [
                    "factor",
                    "distribution",
                    "normalization",
                    "is_sparse",
                    "density",
                    "sparsity",
                    "connectedness",
                    "format",
                    "random_seed",
                    ],
                "bias_kwargs": [
                    "factor",
                    "distribution",
                    "random_seed",
                    ],
                }

        result = {}
        if user is None:
            warnings.warn(f"ESN.__init__: Did not find '{name}' options, using default.")
            result = default.copy()

        else:
            for key, val in user.items():
                if key in allowed[name]:
                    result[key] = val
                else:
                    warnings.warn(f"ESN.__init__: '{key}' not allowed for in {name}, ignoring.")

        return result


    @staticmethod
    def _time_check(array, method, arrayname):
        """Make sure "time" is the last dimension"""

        assert array.dims[-1] == "time", \
                f"{method}: {arrayname} must have 'time' as the final dimension"


    def _get_fcoords(self, dims, coords, n_steps, n_spinup):
        """Get forecast coordinates without the spinup period, and remake a forecast time
        indicating time passed since start of prediction
        This is just for xarray packaging.
        """

        fdims = tuple(d if d != "time" else "ftime" for d in dims)

        tslice = slice(n_spinup, n_spinup+n_steps+1)
        fcoords = {key: coords[key] for key in coords.keys() if key != "time"}
        fcoords["ftime"]= self._get_ftime(coords["time"].isel(time=tslice))

        tvals = coords["time"].isel(time=tslice).values
        fcoords["time"] = xr.DataArray(
            tvals,
            coords={"ftime": fcoords["ftime"]},
            dims="ftime",
            attrs=coords["time"].attrs.copy(),
        )
        return fdims, fcoords


    @staticmethod
    def _get_ftime(time):
        """input time should be sliced to only have initial conditions and prediction"""

        ftime = time.values - time.values[0]

        # handle floating point numbers
        if isinstance(time.data[0], float) and "delta_t" in time.attrs:
            decimals = np.abs(
                Decimal(str(time.delta_t)).as_tuple().exponent
            )
            ftime = np.round(ftime, decimals)

        xftime = xr.DataArray(
                ftime,
                coords={"ftime": ftime},
                dims="ftime",
                attrs={
                    "long_name": "forecast_time",
                    "description": "time passed since prediction initial condition, not including ESN spinup"
                    }
                )
        if "units" in time.attrs:
            xftime.attrs["units"] = time.attrs["units"]
        return xftime


    def _get_attrs(self):
        kw, *_ = inspect.getfullargspec(self.__init__)
        kw.remove("self")

        attrs = dict()
        for key in kw:
            val = getattr(self, key)
            attrs[key] = val
        return attrs


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
