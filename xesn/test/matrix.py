import pytest

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg, sparse

from xesn.matrix import RandomMatrix, SparseRandomMatrix

class TestMatrix:
    n_rows      = 10
    n_cols      = 10
    random_seed = 0


class TestInit(TestMatrix):
    def test_init(self):
        with pytest.raises(AttributeError):
            RandomMatrix(n_rows=self.n_rows, n_cols=self.n_cols, distribution="uniform", blah="nope")

# --- Test distributions from both Dense and Sparse matrices
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

    RM = RandomMatrix

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "random_seed"]}

    def test_dist(self, distribution, error):

        if error is None:
            rm = self.RM(distribution=distribution, **self.kw)
            rm()

        else:
            with pytest.raises(error):
                self.RM(distribution=distribution, **self.kw)


class TestSparseDist(TestDist):
    """This inherits and runs the distribution tests from above"""

    RM      = SparseRandomMatrix
    format  = "csr"
    density = 0.99

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "random_seed", "format", "density"]}


# --- Test normalization
@pytest.mark.parametrize(
        "distribution",
        [ "normal", "uniform" ]
)
@pytest.mark.parametrize(
        "normalization, dense_function, sparse_function, rtol, error",
        [
            ("svd",
                linalg.svdvals,
                lambda x: sparse.linalg.svds(x, k=1, return_singular_vectors=False),
                1e-7,
                None),
            ("eig",
                linalg.eigvals,
                lambda x: sparse.linalg.eigs(x, k=1, return_eigenvectors=False),
                1e-7,
                None),
            ("multiply",
                np.std,
                lambda x: np.std(x.data),
                1e-1,
                None),
            ("spectral_radius", None, None, None, AssertionError),
        ]
)
class TestNorm(TestMatrix):

    RM      = RandomMatrix
    factor  = 10

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "random_seed", "factor"]}

    def test_norm(self, distribution, normalization, dense_function, sparse_function, rtol, error):

        if error is None:
            rm = self.RM(distribution=distribution, normalization=normalization, **self.kw)
            A = rm()
            f = dense_function if not sparse.issparse(A) else sparse_function
            expected = np.max(np.abs(f(A)))
            if distribution != "uniform":
                assert_allclose( self.factor, expected, rtol=rtol )

        else:
            with pytest.raises(error):
                rm = self.RM(distribution=distribution, normalization=normalization, **self.kw)


class TestSparseNorm(TestNorm):
    """This inherits and runs sparse versions of all normalization tests above"""

    RM      = SparseRandomMatrix
    factor  = 10
    format  = "csr"
    density = 0.7

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "random_seed", "factor", "format", "density"]}
