import pytest

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from esnpy.matrix import RandomMatrix



"""
Test for...
- available distributions
- available normalizations
- distribution is what it's specified
- normalization returns what it should be...
"""

class TestMatrix:
    n_rows = 50
    n_cols = 50

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
        "normalization, function, error",
        [
            ("svd", linalg.svdvals, None),
            ("eig", linalg.eigvals, None),
            ("multiply", None, None),
            ("spectral_radius", None, AssertionError),
        ]
)
class TestNorm(TestMatrix):

    distribution    = "uniform"
    factor          = 10

    def test_norm(self, normalization, function, error):

        kw = {"n_rows"          : self.n_rows,
              "n_cols"          : self.n_cols,
              "distribution"    : self.distribution,
              "normalization"   : normalization,
              "factor"          : self.factor}

        if error is None:
            rm = RandomMatrix(**kw)
            A = rm()
            denom = 1
            if function is not None:
                expected = np.max(np.abs(function(A)))

                assert_allclose( self.factor, expected )

        else:
            with pytest.raises(error):
                RandomMatrix(**kw)

