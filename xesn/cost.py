"""Will need to generally do the following

TODO
2. First do NRMSE, spectral will follow.

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

from dask.array import zeros

from .optim import inverse_transform
from .psd import psd

class CostFunction():
    def __init__(self, ESN, train_data, macro_data, config):

        self.ESN        = ESN
        self.train_data = train_data
        self.macro_data = macro_data
        self.config     = config
        if "cost_terms" in config["macro_training"]:
            for key in config["macro_training"]["cost_terms"]:
                try:
                    assert key in ("nrmse", "psd_nrmse")
                except AssertionError:
                    raise NotImplementedError(f"CostFunction.__init__: found '{key}' in config['macro_training']['cost_terms'], only 'nrmse' and 'psd_nrmse' are implemented")


    def __call__(self, macro_param_sets):

        n_parallel, n_params = macro_param_sets.shape

        cost = zeros(n_parallel, chunks=(1,))
        for i, macro_sample in enumerate(macro_param_sets):
            cost[i] = _cost(
                    macro_sample,
                    ESN=self.ESN,
                    train_data=self.train_data,
                    macro_data=self.macro_data,
                    config=self.config)

        # TODO: should we call compute or let the user/driver do that?
        cost = cost.compute()
        return cost.reshape(-1, 1)


def _cost(x_transformed, ESN, train_data, macro_data, config):

    # perform any inverse transformations e.g. of log/log10
    x_names = tuple(config["macro_training"]["parameters"].keys())
    params_transformed = dict(zip(x_names, x_transformed))
    params = inverse_transform(params_transformed, config["macro_training"]["transformations"])

    # update parameters, build, and train esn
    esnc = _update_esn_kwargs(params, config[ESN.__name__.lower()])
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


def nrmse(xds):

    temporal_weights = 1. / xds["truth"].std("ftime")
    norm_error = (xds["prediction"] - xds["truth"]) * temporal_weights
    nmse = (norm_error**2).mean()
    return xp.sqrt(nmse)


def psd_nrmse(xds):
    xds_hat = {}
    for key in ["prediction", "truth"]:
        xds_hat[key] = psd(xds[key])

    return nrmse(xds_hat)


def _update_esn_kwargs(params, original):
    """update the arguments to create ESN based on new parameter dict"""

    esnc = original.copy()

    for key, val in params.items():
        if "_" in key:
            frontend = key[:key.find("_")]
            backend = key[key.find("_")+1:]

            if frontend in ["input", "adjacency", "bias"]:
                esnc[f"{frontend}_kwargs"][backend] = val
            else:
                esnc[key] = val
        else:
            esnc[key] = val

    print(esnc)
    return esnc
