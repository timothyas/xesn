import numpy as np
from copy import deepcopy

def get_samples(xda, n_samples, n_steps, n_spinup, random_seed=None, sample_indices=None):
    """Pull random samples from macro_training or test dataset

    Args:
        xda (xarray.DataArray): with the full chunk of data to pull samples from
        n_samples (int): number of samples to grab
        n_steps (int): number of steps to make in sample prediction
        n_spinup (int): number of spinup steps before prediction
        random_seed (int, optional): RNG seed for grabbing temporal indices of random samples
        samples_indices (list, optional): the temporal indices denoting the start of the prediction period (including spinup), if provided then do not get a random sample of indices first

    Returns:
        samples (list of xarray.DataArray): with each separate sample trajectory
        sample_indices (list of int): with the initial conditions for the start of prediction phase, not the start of spinup
    """

    if sample_indices is None:
        sample_indices = get_sample_indices(
                len(xda["time"]),
                n_samples,
                n_steps,
                n_spinup,
                random_seed)

    else:
        assert len(sample_indices) == n_samples, f"Driver.get_samples: found different values for len(sample_indices) and n_samples"

    samples = []
    for ridx in sample_indices:
        this_sample = xda.isel(time=slice(ridx-n_spinup, ridx+n_steps+1))
        this_sample.attrs["sample_index"] = ridx
        this_sample.attrs["description"] = "sample trajectory from larger dataset, prediction initial conditions start after n_spinup steps"
        this_sample.attrs["n_spinup"] = n_spinup
        this_sample.attrs["n_steps"] = n_steps
        samples.append(this_sample)

    return samples, sample_indices


def get_sample_indices(data_length, n_samples, n_steps, n_spinup, random_seed):
    """Get random sample indices from dataset (without replacement) denoting initial conditions for training, validation, or testing

    Args:
        data_length (int): length of the dataseries along the time dimension
        n_samples (int): number of samples to grab
        n_steps (int): number of steps to make in sample prediction
        n_spinup (int): number of spinup steps before prediction
        random_seed (int, optional): RNG seed for grabbing temporal indices of random samples

    Returns:
        sample_indices (list): with integer indices denoting prediction initial conditions, not start of spinup
    """
    rstate = np.random.RandomState(seed=random_seed)
    n_valid = data_length - (n_steps + n_spinup)
    sample_indices = rstate.choice(n_valid, n_samples, replace=False)

    # add spinup here to get initial condition of the prediction, not including spinup
    sample_indices = list(int(x+n_spinup) for x in sample_indices)
    return sample_indices


def update_esn_kwargs(params, original=None):
    """Update or create a dictionary of keyword arguments used to create an
    :class:`ESN` or :class:`LazyESN` with the values denoted by ``params``.

    Note:
        The ``params`` dict can have ``input_factor`` as its key to update ``original["input_kwargs"]["factor"]`` as is expected by either of the network classes. The same is true for ``adjacency_factor`` and ``bias_factor``.

    Args:
        params (dict): parameter names and values contain either array/list or value of parameter
        original (dict, optional): with options used to create an :class:`ESN` or :class:`LazyESN`, if not provided then create a new dictionary
    Returns:
        esnc (dict): with the updated or new options based on ``params``
    """

    esnc = dict() if original is None else deepcopy(original)

    for key, val in params.items():
        # do this for nicer yaml dumping
        valnice = float(val) if isinstance(val, float) else val
        if "_" in key:
            frontend = key[:key.find("_")]
            backend = key[key.find("_")+1:]

            if frontend in ["input", "adjacency", "bias"]:
                fkw = f"{frontend}_kwargs"
                if fkw in esnc:
                    esnc[fkw][backend] = valnice
                else:
                    esnc[fkw] = {backend: valnice}

            else:
                esnc[key] = valnice
        else:
            esnc[key] = valnice

    return esnc
