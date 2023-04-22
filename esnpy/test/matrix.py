import pytest

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from esnpy.matrix import RandomMatrix, SparseRandomMatrix



"""
Test for...
- available distributions
- available normalizations
- distribution is what it's specified
- normalization returns what it should be...
"""

class TestMatrix:
    n_rows = 10
    n_cols = 10


class TestInit(TestMatrix):
    def test_init(self):
        with pytest.raises(AttributeError):
            RandomMatrix(n_rows=self.n_rows, n_cols=self.n_cols, distribution="uniform", blah="nope")

@pytest.mark.parametrize(
        "distribution, error",
        [
            ("uniform", None),
            ("gaussian", None),
            ("normal", None),
            ("gamma", AssertionError),
        ],
)
class TestDist(TestMatrix):

    def test_dist(self, distribution, error):

        kw = {"n_rows"          : self.n_rows,
              "n_cols"          : self.n_cols,
              "distribution"    : distribution}

        if error is None:
            rm = RandomMatrix(**kw)
            rm()

        else:
            with pytest.raises(error):
                RandomMatrix(**kw)

@pytest.mark.parametrize(
        "distribution",
        [ "normal", "uniform" ]
)
@pytest.mark.parametrize(
        "normalization, function, rtol, error",
        [
            ("svd", linalg.svdvals, 1e-7, None),
            ("eig", linalg.eigvals, 1e-7, None),
            ("multiply", np.std, 1e-2, None),
            ("spectral_radius", None, None, AssertionError),
        ]
)
class TestNorm(TestMatrix):

    factor          = 10
    random_seed     = 0

    def test_norm(self, distribution, normalization, function, rtol, error):

        kw = {"n_rows"          : self.n_rows,
              "n_cols"          : self.n_cols,
              "distribution"    : distribution,
              "normalization"   : normalization,
              "factor"          : self.factor,
              "random_seed"     : self.random_seed}

        if error is None:
            rm = RandomMatrix(**kw)
            A = rm()
            expected = np.max(np.abs(function(A)))
            assert_allclose( self.factor, expected, rtol=rtol )

        else:
            with pytest.raises(error):
                RandomMatrix(**kw)
