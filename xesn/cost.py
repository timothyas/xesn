"""Will need to generally do the following

Order of operations
1. Do this for lazy, then worry about generalizing.
2. First do NRMSE, spectral will follow.
3. Address temporal weights (for nrmse, not long term mean)

TODOs
- config vs parameters ... this is getting confusing even to me
- separate parameters involved in optimization:
    - EGO
    - validation/optim forecasts
    - data management like persist
- Do we need to _check_config like with driver? or assume it's coming from driver?
- Should this inherit driver?
- Right now, computing temporal normalization on the fly for each macro sample... not how we did it before... does this matter?
- is it faster to make dask.array.zeros containers, or a list that gets .compute called on it?
- is 1e9 a reasonable cost threshold?

Other optim stuff
1. get the training sets, same as testing.
2. get the optim bounds and argument data types
3. do the optimization (passing sampling args to the EGO, rather than having a separate function to do it)
4. map log/log10 back to parameters
"""

from dask.array import zeros

class CostFunction():
    def __init__(self, ESNModel, micro_data, macro_data, config):

        self.ESN        = ESNModel
        self.micro_data = micro_data
        self.macro_data = macro_data
        self.config     = config


    def __call__(macro_param_sets, micro_data, macro_data):

        #TODO: figure out what size cost needs to be again...
        # For now assuming that macro_param_sets is this shape:
        n_parallel, n_params = macro_param_sets.shape

        cost = zeros(n_parallel, chunks=(1,))
        for i, macro_sample in enumerate(macro_params_sets):
            cost[i] = _cost(
                    macro_sample,
                    ESNModel=self.ESNModel,
                    micro_data=self.micro_data,
                    macro_data=self.macro_data,
                    config=self.config)

        cost = cost.compute()
        return cost.reshape(-1, 1)


def _cost(self, macro_params, ESNModel, micro_data, macro_data, config):

    macro_names = tuple(config["optim"]["parameters"].keys())

    esnc = _get_esn_args(macro_params, macro_names, config[ESNModel.__name__.lower()])

    esn = ESNModel(**esnc)
    esn.build()
    esn.train(micro_data, **config["training"])

    n_macro = config["optim"]["n_samples"]
    n_spinup = config["optim"]["n_spinup"]
    n_steps = config["optim"]["n_forecast_steps"]

    cost = zeros(n_macro, chunks=(1,))
    for i, truth in enumerate(macro_data):
        xds = esn.test(truth, n_steps=n_steps, n_spinup=n_spinup)
        cost[i] = nrmse(xds)

    avg_cost = cost.mean()

    if xp.isnan(avg_cost) or xp.isinf(avg_cost) or avg_cost > 1.e9:
        avg_cost = 1.e9
    else:
        avg_cost = avg_cost.persist() if config["optim"]["persist"] else avg_cost
    return avg_cost


def nrmse(xds):

    # right now just compute NRMSE, will separate this later
    temporal_weights = 1. / xds["truth"].std("time")
    norm_error = (xds["prediction"] - xds["truth"]) * temporal_weights
    nmse = (norm_error**2).mean()
    return xp.sqrt(nmse)


def _get_esn_args(macro_params, macro_names, config):

    esnc = config.copy()
    for key, val in zip(macro_names, macro_params):
        if key[:5] == "log10":
            esnc[key[6:]] = 10. ** val
        elif key[:3] == "log":
            esnc[key[4:]] = xp.exp( val )
        else:
            esnc[key] = val

    return esnc
