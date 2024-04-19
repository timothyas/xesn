"""Will need to generally do the following

Questions
- Do we need to _check_config like with driver? or assume it's coming from driver?
- Right now, computing temporal normalization on the fly for each macro sample... not how we did it before... does this matter?
- is it faster to make dask.array.zeros containers, or a list that gets .compute called on it?
- is 1e9 a reasonable cost threshold?
"""
from . import _use_cupy
if _use_cupy:
    import cupy as xp

else:
    import numpy as xp

import xarray as xr
from dask.array import zeros

from .optim import inverse_transform
from .psd import psd
from .utils import update_esn_kwargs

class CostFunction():
    """A class used to evaluate an ESN architecture, which can be used in :func:`xesn.optimize`.
    See `this page of the documentation  <https://xesn.readthedocs.io/en/latest/example_macro_training.html>`_ for example usage.
    Currently the following generic cost function is implemented

    .. math::
	    \mathcal{J}_\\text{macro}(\mathbf{\\theta}) = \dfrac{1}{N_\\text{macro}}\sum_{j=1}^{N_\\text{macro}} \left\{\gamma_1\\text{NRMSE}(j) + \gamma_2 \\text{PSD}\_\\text{NRMSE}(j)\\right\}

    .. math::
        \\text{NRMSE}(j) = \sqrt{\dfrac{1}{N_v N_\\text{steps}}\sum_{n=1}^{N_\\text{steps}}\sum_{i=1}^{N_v}\left(\dfrac{\hat{v}_j(i, n) - v_j(i, n)}{SD_j}\\right)^2 }

    .. math::
	    \\text{PSD}\_\\text{NRMSE}(j) = \sqrt{\dfrac{1}{N_K N_\\text{steps}}\sum_{n=1}^{N_\\text{steps}}\sum_{i=1}^{N_K}\left(\dfrac{\hat{\psi}_j(k, n) - \psi_j(k, n)}{SD_j(k)}\\right)^2 }

    where:

    - :math:`\mathbf{\\theta}` is our vector of parameters to be optimized, defined by ``config["macro_training"]["parameters"]``

    - :math:`N_\\text{macro}` = ``config["macro_training"]["forecast"]["n_samples"]`` is the number of sample forecasts

    - :math:`\gamma_1` and :math:`\gamma_2` determine the overall weighting for the Normalized Root Mean Square Error (NRMSE) and Power Spectral Density NRMSE (PSD_NRMSE) terms, respectively. These are controlled with ``config["macro_training"]["cost_terms"]``

    - :math:`i` is the index for each non-time index

    - :math:`n` is the temporal index, and :math:`N_\\text{steps}` = ``config["macro_training"]["forecast"]["n_steps"]`` is the length of each sample forecast in terms of the number of time steps

    - :math:`j` is the index for each sample forecast

    - :math:`k` is the index for each spectral mode of the PSD

    - :math:`\psi_j(k,n)` is the :math:`k^{th}` mode's amplitude, for sample :math:`j` at time step :math:`n`

    - The standard deviation used in the NRMSE calculation is

    .. math::
        SD_j = \sqrt{\dfrac{\sum_{i=1}^{N_v}\sum_{n=1}^{N_{\\text{steps}}}\left(v_j(i, n) - \mu_j\\right)^2}{(N_{\\text{steps}}-1)(N_v-1)}}

    - :math:`SD_j(k)` used in the PSD_NRMSE calculation is defined similarly as above, but in spectral space, and note that each mode is normalized separately as different modes can vary by vastly different orders of magnitude

    - :math:`\mu_j` is the average taken over space and time

    .. math::
        \mu_j = \dfrac{1}{N_v N_\\text{steps}} \sum_{n=1}^{N_\\text{steps}} \sum_{i=1}^{N_v} v_j(i,n)

    - the average used for PSD_NRMSE :math:`\mu_j(k)` is similarly defined for PSD except the summation is only taken over time

    Args:
        ESN (:class:`ESN` or :class:`LazyESN`): the ESN class to be used
        train_data (xarray.DataArray): containing the training data used train the readout weights :math:`\mathbf{W}_\\text{out}`
        macro_data (List[xarray.DataArray]): containing the sample trajectories to compute the cost function with
        config (dict): with sections ``"macro_training"``, ``"training"``, ``esn`` or ``lazyesn`` depending on which class is used, and optionally ``"testing"`` in order to use :meth:`evaluate` on the test data. See `here <https://xesn.readthedocs.io/en/latest/example_macro_training.html#CostFunction-and-Optimization-Configuration>`_ for an example on how to set this up.
        test_data (xarray.DataArray, optional): separate test data passed here for convenience to be used with :meth:`evaluate`
	"""

    __slots__ = ("ESN", "train_data", "macro_data", "config", "test_data")

    def __init__(self, ESN, train_data, macro_data, config, test_data=None):

        self.ESN        = ESN
        self.train_data = train_data
        self.macro_data = macro_data
        self.test_data  = test_data
        self.config     = config
        if "cost_terms" in config["macro_training"]:
            for key in config["macro_training"]["cost_terms"]:
                try:
                    assert key in ("nrmse", "psd_nrmse")
                except AssertionError:
                    raise NotImplementedError(f"CostFunction.__init__: found '{key}' in config['macro_training']['cost_terms'], only 'nrmse' and 'psd_nrmse' are implemented")


    def __call__(self, macro_param_sets, is_transformed=True):
        """Evaluate the cost function for a given set of parameter values

        Args:
            macro_param_sets (array_like): with shape ``n_sets x n_parameters``, where ``n_parameters`` is the number of scalar parameters being optimized, ``n_sets`` would be the number of examples to evaluate
            is_transformed (bool, optional): if True, then any transformations specified by ``config["macro_training"]["transformations"]`` have been applied. See :func:`xesn.optimize.transform` for an example

        Returns:
            cost (array_like): with shape ``n_sets x 1``, cost evaluate for each parameter set
        """

        macro_param_sets = xp.atleast_2d(macro_param_sets)

        n_parallel, n_params = macro_param_sets.shape

        cost = zeros(n_parallel, chunks=(1,))
        for i, macro_sample in enumerate(macro_param_sets):

            cost[i] = _cost(
                    macro_sample,
                    ESN=self.ESN,
                    train_data=self.train_data,
                    macro_data=self.macro_data,
                    config=self.config,
                    is_transformed=is_transformed)

        # TODO: should we call compute or let the user/driver do that?
        cost = cost.compute()
        return cost.reshape(-1, 1)


    def evaluate(self, parameters, use_test_data=False):
        """Evaluate this parameter set by building an ESN and testing it on either the :attr:`macro_data` or :attr:`test_data`.
        See `this page of the documentation  <https://xesn.readthedocs.io/en/latest/example_macro_training.html>`_ for example usage.

        Args:
            parameters (dict): with keys as the parameter names, and values as the parameter values
            use_test_data (bool, optional): if True, use the :attr:`test_data`, otherwise use :attr:`macro_data` to evaluate this parameter set
        """

        # setup and train ESN
        kwargs = update_esn_kwargs(parameters, self.config[self.ESN.__name__.lower()])
        esn = self.ESN(**kwargs)
        esn.build()
        esn.train(self.train_data, **self.config["training"])

        # run the forecasts
        if not use_test_data:
            datasets = self.macro_data
            n_spinup = self.config["macro_training"]["forecast"]["n_spinup"]
            n_steps = self.config["macro_training"]["forecast"]["n_steps"]


        else:
            datasets = self.test_data
            n_spinup = self.config["testing"]["n_spinup"]
            n_steps = self.config["testing"]["n_steps"]


        dslist = []
        for i, truth in enumerate(datasets):
            result = esn.test(truth, n_steps=n_steps, n_spinup=n_spinup)

            if "nrmse" in self.config["macro_training"]["cost_terms"]:
                result["nrmse"] = nrmse(result, drop_time=False)

            if "psd_nrmse" in self.config["macro_training"]["cost_terms"]:
                result["psd_truth"] = psd(result["truth"])
                result["psd_prediction"] = psd(result["prediction"])
                result["psd_nrmse"] = nrmse({
                    "truth": result["psd_truth"],
                    "prediction": result["psd_prediction"],
                }, drop_time=False)

            result = result.expand_dims({"sample": [i]})
            dslist.append(result)

        xds = xr.concat(dslist, dim="sample")
        return xds


def _cost(x, ESN, train_data, macro_data, config, is_transformed=True):

    # perform any inverse transformations e.g. of log/log10
    x_names = tuple(config["macro_training"]["parameters"].keys())
    if is_transformed:
        params_transformed = dict(zip(x_names, x))
        params = inverse_transform(
            params_transformed,
            config["macro_training"]["transformations"]
        )

    else:
        params = dict(zip(x_names, x))

    # update parameters, build, and train esn
    esnc = update_esn_kwargs(params, config[ESN.__name__.lower()])
    esn = ESN(**esnc)
    esn.build()
    esn.train(train_data, **config["training"])

    # run the forecasts to compute cost
    n_macro = config["macro_training"]["forecast"]["n_samples"]
    n_spinup = config["macro_training"]["forecast"]["n_spinup"]
    n_steps = config["macro_training"]["forecast"]["n_steps"]
    terms = config["macro_training"].get("cost_terms", {"nrmse": 1.})

    cost = zeros(n_macro, chunks=(1,))
    for i, truth in enumerate(macro_data):
        xds = esn.test(truth, n_steps=n_steps, n_spinup=n_spinup)
        all_costs = []
        for key, factor in terms.items():
            if key == "nrmse":
                all_costs.append( factor * nrmse(xds).data )
            elif key == "psd_nrmse":
                all_costs.append( factor * psd_nrmse(xds).data )

        cost[i] = xp.sum(all_costs)

    avg_cost = cost.mean()

    # Deal with persist here
    #avg_cost = avg_cost.persist() if config["optim"]["persist"] else avg_cost

    # Deal with wacky numbers
    default = 1e9
    cost_upper_bound = config["macro_training"].get("cost_upper_bound", default)
    cost_upper_bound = default if cost_upper_bound is None else cost_upper_bound
    if xp.isnan(avg_cost) or xp.isinf(avg_cost) or avg_cost > cost_upper_bound:
        avg_cost = cost_upper_bound
    return avg_cost


def nrmse(xds, drop_time=True):
    """Compute the NRMSE between truth and prediction

    Args:
        xds (xarray.Dataset): with data_vars ``"prediction"`` and ``"truth"`` to be compared
        drop_time (bool, optional): if True, take the average also over time, otherwise compute NRMSE evolving over time

    Returns:
        nrmse (xarray.DataArray): with the normalized root mean square error
    """

    time = "ftime" if "ftime" in xds["truth"].dims else "time"

    temporal_weights = 1. / xds["truth"].std(time)
    norm_error = (xds["prediction"] - xds["truth"]) * temporal_weights

    if drop_time:
        dims = norm_error.dims
    else:
        dims = tuple(d for d in norm_error.dims if d != time)

    nmse = (norm_error**2).mean(dim=dims)
    return xr.DataArray(
        xp.sqrt(nmse.data),
        coords=nmse.coords,
        dims=nmse.dims,
        attrs=nmse.attrs.copy(),
    )


def psd_nrmse(xds, drop_time=True):
    """Compute the NRMSE of the PSD

    Args:
        xds (xarray.Dataset): with data_vars ``"prediction"`` and ``"truth"`` to be compared
        drop_time (bool, optional): if True, take the average also over time, otherwise compute NRMSE evolving over time

    Returns:
        psd_nrmse (xarray.DataArray): with the normalized root mean square error of the power spectral density
    """

    xds_hat = {}
    for key in ["prediction", "truth"]:
        xds_hat[key] = psd(xds[key])

    return nrmse(xds_hat, drop_time=drop_time)
