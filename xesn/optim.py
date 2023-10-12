"""

TODO
----
- do we want to transform the bounds? is that confusing? should this be specified with a "transform" option?
- look for initial sample and remap between named dictionary and array values
"""
from smt.applications import EGO
from smt.surrogate_models import KRG, XSpecs
from smt.utils.design_space import DesignSpace

def optimize(self, macro_params, transformations, cost_function, **kwargs):
    """
    Args:
        macro_params (dict): containing parameter name and bounds as key, val pairs
        transformations (dict): containing parameter name and transformation (e.g., log10) as key, val pairs
        cost_function: a created CostFunction obect
        **kwargs: passed to EGO
    """

    bounds_transformed  = transform(macro_params, transformations)
    design_space        = DesignSpace(bounds_transformed)
    surrogate           = KRG(design_space=design_space, print_global=False)

    ego = EGO(surrogate=surrogate, **kwargs)
    x_opt_transformed, y_opt, _, x_data, y_data = ego.optimize(fun=cost_function)

    param_opt_transformed = zip(macro_params.keys(), x_opt_transformed)
    param_opt = inverse_transform(opt_transformed, transformations)

    print("\n --- Optimization Results ---")
    print("\nOptimal inputs:")
    for key, val in param_opt.items():
        print(f"\t{key:<28s}: {val}")

    print("\nApproximate cost minimum:")
    print(f"\t{y_opt}\n")

    return param_opt


def transform(params, transformations):
    """
    Args:
        parameters (dict): parameter names and values contain either array/list or value of parameter
        transformations (dict): with what we want to do to each variable for optimization, e.g. log

    Returns:
        transformed_params
    """

    transformed_params = params.copy()
    for key, transform in transformations.items():

        try:
            assert key in params
        except AssertionError:
            raise KeyError(f"Could not find {key} in optimization parameters, was provided {tuple(params.keys())}")

        try:
            assert transform in ("log", "log10")
        except AssertionError:
            raise NotImplementedError(f"transformation {transform} unrecognized for parameter {key}, only 'log' and 'log10' implemented")

        # Note that I'm not renaming the parameters in order to keep the dict order
        if transform == "log10":
            transformed_params[key] = _transform(xp.log10, val)

        elif transform == "log":
            transformed_params[key] = _transform(xp.log, val)

        return transformed_params


def inverse_transform(transformed_params, transformations):

    params = transformed_params.copy()
    for key, transform in transformations.items():

        try:
            assert key in params
        except AssertionError:
            raise KeyError(f"Could not find {key} in optimization parameters, was provided {tuple(params.keys())}")

        try:
            assert transform in ("log", "log10")
        except AssertionError:
            raise NotImplementedError(f"transformation {transform} unrecognized for parameter {key}, only 'log' and 'log10' implemented")

        if transform == "log10":
            params[key] = _transform(lambda x : 10. ** x, val)

        elif transform == "log":
            params[key] = _transform(xp.exp, val)

        return params



def _transform(fun, val):
    val = params[key]
    if isinstance(val, (float, int)):
        newval = fun(val)
    else:
        newval = [fun(x) for x in val]

    return newval
