
from scipy import stats

from . import _use_cupy
if _use_cupy:
    import cupy as xp
    from cupy import linalg
    from cupyx.scipy import sparse
    import cupyx.scipy.sparse.linalg # gives access to sparse.linalg

else:
    import numpy as xp
    from scipy import sparse, linalg

class RandomMatrix():

    # Set by user
    n_rows          = None
    n_cols          = None

    distribution    = None
    normalization   = "multiply"
    factor          = 1.0

    random_seed     = None

    # Set automatically
    dist_kw         = None

    def __init__(self, n_rows, n_cols, distribution, **kw):

        self.n_rows         = n_rows
        self.n_cols         = n_cols
        self.distribution   = distribution

        # fill in optional attributes
        for key, val in kw.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        # Check inputs
        assert self.distribution in ("uniform", "gaussian", "normal")
        assert self.normalization in ("svd", "eig", "multiply")

        # Create random state
        self.random_state = xp.random.RandomState(self.random_seed)


    def __call__(self):
        A = self.create_matrix()
        return self.normalize(A)


    def create_matrix(self):

        if self.distribution == "uniform":
            A = self.random_state.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(self.n_rows, self.n_cols))
        else:
            A = self.random_state.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(self.n_rows, self.n_cols))

        return A


    def normalize(self, A):

        denominator = 1.0
        if self.normalization == "svd":
            s = linalg.svd(A, compute_uv=False, full_matrices=False)
            denominator = xp.max(xp.abs(s))

        elif self.normalization == "eig":
            s = linalg.eigvals(A)
            denominator = xp.max(xp.abs(s))


        return self.factor / denominator * A


class SparseRandomMatrix(RandomMatrix):

    density         = None
    format          = None

    def __init__(self, n_rows, n_cols, distribution, density, format, **kw):

        super().__init__(
                n_rows=n_rows,
                n_cols=n_cols,
                distribution=distribution,
                **kw)

        self.density    = density
        self.format     = format


    def create_matrix(self):

        if self.distribution == "uniform":
            distribution = self.random_state.rand

        else:
            distribution = self.random_state.randn

        A = sparse.random(
            self.n_rows,
            self.n_cols,
            density=self.density,
            format=self.format,
            random_state=self.random_state,
            data_rvs=distribution)

        return A


    def normalize(self, A):

        denominator = 1.0
        if self.normalization == "svd":
            s = sparse.linalg.svds(A, k=1, which="LM", return_singular_vectors=False)
            denominator = xp.max(xp.abs(s))

        elif self.normalization == "eig":
            s = sparse.linalg.eigs(A, k=1, which="LM", return_eigenvectors=False)
            denominator = xp.max(xp.abs(s))

        A.data = self.factor / denominator * A.data
        return A

