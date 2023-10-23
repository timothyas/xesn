import pytest

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg, sparse

from xesn.matrix import RandomMatrix, SparseRandomMatrix

class TestMatrix:
    n_rows          = 10
    n_cols          = 10
    factor          = 1.0
    normalization   = "multiply"
    random_seed     = 0


# --- Test distributions from both Dense and Sparse matrices
@pytest.mark.parametrize(
        "distribution, error",
        [
            ("uniform", None),
            ("gaussian", None),
            ("normal", None),
            ("gamma", NotImplementedError),
        ],
)
class TestDist(TestMatrix):

    RM = RandomMatrix

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "factor", "normalization", "random_seed"]}

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
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "factor", "normalization", "random_seed", "format", "density"]}


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
            ("spectral_radius", None, None, None, NotImplementedError),
        ]
)
class TestNorm(TestMatrix):

    RM      = RandomMatrix
    factor  = 10

    @property
    def kw(self):
        return {key: getattr(self, key) for key in ["n_rows", "n_cols", "factor", "random_seed", "factor"]}

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

@pytest.mark.parametrize(
        "n_cols, density, sparsity, connectedness, error",
        [
            (10, 0.1, None, None, None),
            (10, None, 0.9, None, None),
            (10, None, None, 1, None),
            (10, None, None, None, TypeError),
            (10, 0.1, 0.9, None, TypeError),
            (10, 0.1, None, 1, TypeError),
            (10, None, 0.9, 1, TypeError),
            (5,  0.1, None, None, None),
            (5,  None, 0.9, None, None),
            (5,  None, None, 1, TypeError),
        ]
    )
def test_sparse_mat_inputs(n_cols, density, sparsity, connectedness, error):
    """test the whole density, sparsity, connectivity stuff"""

    if error is None:
        sm = SparseRandomMatrix(
                n_rows=10,
                n_cols=n_cols,
                factor=1.0,
                distribution="normal",
                normalization="multiply",
                density=density,
                sparsity=sparsity,
                connectedness=connectedness,
                )
        sm()

        if n_cols == 10:
            assert_allclose(sm.density, 0.1)
    else:
        with pytest.raises(error):
            sm = SparseRandomMatrix(
                    n_rows=10,
                    n_cols=n_cols,
                    factor=1.0,
                    distribution="normal",
                    normalization="multiply",
                    density=density,
                    sparsity=sparsity,
                    connectedness=connectedness,
                    )
