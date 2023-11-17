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
        """

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
        xds["nrmse"] = nrmse(xds, drop_time=False)
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

    time = "ftime" if "ftime" in xds["truth"].dims else "time"

    temporal_weights = 1. / xds["truth"].std(time)
    norm_error = (xds["prediction"] - xds["truth"]) * temporal_weights

    if drop_time:
        dims = norm_error.dims
    else:
        dims = tuple(d for d in norm_error.dims if d != time)

    nmse = (norm_error**2).mean(dim=dims)
    return xp.sqrt(nmse)


def psd_nrmse(xds):
    xds_hat = {}
    for key in ["prediction", "truth"]:
        xds_hat[key] = psd(xds[key])

    return nrmse(xds_hat)
