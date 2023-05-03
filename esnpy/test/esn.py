import pytest

import numpy as np
from numpy.testing import assert_allclose
from esnpy.esn import ESN

class TestESN:
    n_input             = 3
    n_output            = 1
    n_reservoir         = 100
    connectedness       = 5
    bias                = .1
    leak_rate           = 0.5
    tikhonov_parameter  = 1e-6
    random_seed         = 0

    input_factor        = 0.5
    adjacency_factor    = 0.9


class TestInit(TestESN):

    @property
    def kw(self):
        return {key: getattr(self, key) for key in [
            "n_input", "n_output", "n_reservoir", "connectedness", "bias", "leak_rate", "tikhonov_parameter", "input_factor", "adjacency_factor", "random_seed"]}


    def test_basic(self):
        esn = ESN(**self.kw)
        print(esn)
        for key, expected in self.kw.items():
            assert_allclose(getattr(esn, key), expected)


    @pytest.mark.parametrize(
            "key, val, raises, error",
            [
                ("bias", -1., pytest.raises, ValueError),
                ("connectedness", 101, pytest.raises, ValueError),
                ("connectedness", 90, pytest.warns, RuntimeWarning),
            ],
    )
    def test_errors(self, key, val, raises, error):
        kwargs = self.kw.copy()
        kwargs[key] = val
        with raises(error):
            esn = ESN(**kwargs)
