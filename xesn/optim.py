from . import _use_cupy
if _use_cupy:
    import cupy as xp

else:
    import numpy as xp

from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.utils.design_space import DesignSpace

def optimize(cost_function, **kwargs):
    """A simple interface with `EGO <https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html>`_ to perform Bayesian Optimization and solve for an optimal parameter set.
    See `this page of the documentation  <https://xesn.readthedocs.io/en/latest/example_macro_training.html>`_ for example usage.

    Args:
        cost_function: a created CostFunction obect
        **kwargs: passed to EGO, see possible options `here <https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html#options>`_

    Returns:
        params_opt (dict): with keys as parameter names and values as the determined optimal parameter values
    """

    if _use_cupy:
        raise NotImplementedError(f"optimization uses smt, which does not support GPUs/cupy")

    macro_params = cost_function.config["macro_training"]["parameters"]
    transformations = cost_function.config["macro_training"]["transformations"]
    bounds_transformed  = transform(macro_params, transformations)
    design_space        = DesignSpace(list(bounds_transformed.values()))
    surrogate           = KRG(design_space=design_space, print_global=False)

    ego = EGO(surrogate=surrogate, **kwargs)
    x_opt_transformed, y_opt, *_ = ego.optimize(fun=cost_function)

    params_opt_transformed = dict(zip(macro_params.keys(), x_opt_transformed))
    params_opt = inverse_transform(params_opt_transformed, transformations)

    print("\n --- Optimization Results ---")
    print("\nOptimal inputs:")
    for key, val in params_opt.items():
        print(f"\t{key:<28s}: {val}")

    print("\nApproximate cost minimum:")
    print(f"\t{y_opt}\n")

    return params_opt


def transform(params, transformations):
    """Transform parameters for optimization, with only either log or log10.
    Parameters with unspecified transformations are untouched.

    Args:
        parameters (dict): parameter names and values contain either array/list or value of parameter
        transformations (dict): with what we want to do to each variable for optimization, e.g. log

    Returns:
        transformed_params (dict): with updated parameters based on transformations, or untouched if not specified

    Example:
        >>> params = {"input_factor": 0.5, "adjacency_factor": 0.5, "bias_factor": 0.5}
        >>> transforms = {"input_factor": "log", "adjacency_factor": "log10"}
        >>> transform(params, transforms)
        {'input_factor': -0.6931471805599453,
         'adjacency_factor': -0.3010299956639812,
         'bias_factor': 0.5}
    """

    transformed_params = params.copy()
    for key, transform in transformations.items():

        try:
            assert key in params
        except AssertionError:
            raise KeyError(f"Could not find '{key}' in optimization parameters, was provided {tuple(params.keys())}")

        try:
            assert transform in ("log", "log10")
        except AssertionError:
            raise NotImplementedError(f"transformation {transform} unrecognized for parameter {key}, only 'log' and 'log10' implemented")

        # Note that I'm not renaming the parameters in order to keep the dict order
        val = params[key]
        if transform == "log10":
            transformed_params[key] = _transform(xp.log10, val)

        elif transform == "log":
            transformed_params[key] = _transform(xp.log, val)

    return transformed_params


def inverse_transform(transformed_params, transformations):
    """Perform the inverse of the specified transformation, of either only log or log10.
    Parameters with unspecified transformations are untouched.

    Args:
        transformed_parameters (dict): parameter names and values contain either array/list or value of parameter
        transformations (dict): with what we want to do to each variable for optimization, either log or log10 (or unspecified)

    Returns:
        transformed_params (dict): with updated parameters based on transformations, or untouched if not specified

    Example:
        >>> params = {"input_factor": -0.69, "adjacency_factor": -0.3, "bias_factor": 0.5}
        >>> transforms = {"input_factor": "log", "adjacency_factor": "log10"}
        >>> inverse_transform(params, transforms)
        {'input_factor': 0.5015760690660556,
         'adjacency_factor': 0.5011872336272722,
         'bias_factor': 0.5}
    """

    params = transformed_params.copy()
    for key, transform in transformations.items():

        try:
            assert key in params
        except AssertionError:
            raise KeyError(f"Could not find '{key}' in optimization parameters, was provided {tuple(params.keys())}")

        try:
            assert transform in ("log", "log10")
        except AssertionError:
            raise NotImplementedError(f"transformation {transform} unrecognized for parameter {key}, only 'log' and 'log10' implemented")

        val = params[key]
        if transform == "log10":
            params[key] = _transform(lambda x : 10. ** x, val)

        elif transform == "log":
            params[key] = _transform(xp.exp, val)

    return params


def _transform(fun, val):
    if isinstance(val, (float, int)):
        newval = fun(val)
    else:
        newval = [fun(x) for x in val]

    return newval
