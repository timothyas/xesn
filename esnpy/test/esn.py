import pytest

import numpy as np
from numpy.testing import assert_allclose
from esnpy.esn import ESN

class TestESN:
    n_input             = 3
    n_output            = 1
    n_reservoir         = 100
    connectedness       = 5
    bias                = 0.1
    leak_rate           = 0.5
    tikhonov_parameter  = 1e-6

    input_factor        = 0.1
    adjacency_factor    = 0.1
    @property
    def kw(self):
        return {key: getattr(self, key) for key in [
            "n_input", "n_output", "n_reservoir", "connectedness", "bias", "leak_rate", "tikhonov_parameter", "input_factor", "adjacency_factor"]}


class TestInit(TestESN):

    def test_basic(self):
        esn = ESN(**self.kw)
        str(esn)
        assert esn.__repr__() == str(esn)
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

@pytest.mark.parametrize(
        "distribution", ["uniform", "normal", "gaussian"],
)
@pytest.mark.parametrize(
        "factor, error", [(0.1, None), (0.5, ValueError)],
)
class TestVectors(TestESN):

    def test_bias_kwargs(self, distribution, factor, error):

        bkw = { "distribution"  : distribution,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if error is None:
            esn = ESN(**self.kw, bias_kwargs=bkw)
            for key, expected in bkw.items():
                assert esn.bias_kwargs[key] == expected

            esn.build()
            assert len(esn.bias_vector) == self.n_reservoir

        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, bias_kwargs=bkw)

@pytest.mark.parametrize(
        "distribution", ["uniform", "normal", "gaussian"],
)
@pytest.mark.parametrize(
        "factor, error", [(0.1, None), (0.5, ValueError)],
)
@pytest.mark.parametrize(
        "is_sparse", [True, False],
)
class TestMatrices(TestESN):

    @pytest.mark.parametrize(
            "normalization", ["multiply", "svd"],
    )
    def test_input_kwargs(self, distribution, normalization, is_sparse, factor, error):
        ikw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if is_sparse:
            ikw["density"] = 0.1

        if error is None:
            esn = ESN(**self.kw, input_kwargs=ikw)
            for key, expected in ikw.items():
                assert esn.input_kwargs[key] == expected

            esn.build()
            assert tuple(esn.Win.shape) == (self.n_reservoir, self.n_input)
        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, input_kwargs=ikw)


    @pytest.mark.parametrize(
            "normalization", ["multiply", "eig", "svd"],
    )
    def test_adjacency_kwargs(self, distribution, normalization, is_sparse, factor, error):
        akw = { "distribution"  : distribution,
                "normalization" : normalization,
                "is_sparse"     : is_sparse,
                "factor"        : factor,
                "random_seed"   : 0,
                }
        if is_sparse:
            akw["density"] = 0.1

        if error is None:
            esn = ESN(**self.kw, adjacency_kwargs=akw)
            for key, expected in akw.items():
                assert esn.adjacency_kwargs[key] == expected

            esn.build()
            assert tuple(esn.W.shape) == (self.n_reservoir, self.n_reservoir)
        else:
            with pytest.raises(error):
                esn = ESN(**self.kw, adjacency_kwargs=akw)


# Test training
# all with or without spinup
# all with or without batch_size
# check for spinup <= time assertion
# check that we can use batch size bigger than data length

class TestTraining(TestESN):
    n_train = 500

    def test_simple(self):
        """where input = output, no other options"""
        kw = self.kw.copy()
        kw["n_output"] = self.n_input
        u = np.random.normal(size=(self.n_input, self.n_train))
        esn = ESN(**kw)
        esn.build()
        esn.train(u)


    @pytest.mark.parametrize(
            "n_input, n_output",  [(3, 7), (3, 3)],
    )
    @pytest.mark.parametrize(
            "n_spinup", [0, 10],
    )
    @pytest.mark.parametrize(
            "batch_size", [None, 33],
    )
    def test_all_options(self, n_input, n_output, n_spinup, batch_size):
        u = np.random.normal(size=(n_input, self.n_train))
        y = np.random.normal(size=(n_output, self.n_train))

        kwargs = self.kw.copy()
        kwargs["n_input"] = n_input
        kwargs["n_output"] = n_input
        esn = ESN(**kwargs)
        esn.build()
        esn.train(u, y=y, n_spinup=n_spinup, batch_size=batch_size)

        assert tuple(esn.Wout.shape) == (n_output, self.n_reservoir)

    def test_spinup_assert(self):
        kw = self.kw.copy()
        kw["n_output"] = self.n_input
        u = np.random.normal(size=(self.n_input, self.n_train))
        esn = ESN(**kw)
        esn.build()
        with pytest.raises(AssertionError):
            esn.train(u, n_spinup=self.n_train+1)

