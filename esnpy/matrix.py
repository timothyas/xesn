
from scipy import stats

try:
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()
    from cupy import linalg
    from cupyx.scipy import sparse
    import cupyx.scipy.sparse.linalg # gives access to sparse.linalg

except xp.cuda.runtime.CUDARuntimeError:
    import numpy as xp
    from scipy import sparse, linalg

class RandomMatrix():

    # Set by user
    is_symmetric    = False
    is_antisymmetric= False

    n_rows          = None
    n_cols          = None

    distribution    = None
    normalization   = None
    factor          = 1.0

    random_state    = None

    # Set automatically
    dist_kw         = None

    def __init__(self, **kw):

        # fill those attributes
        for key, val in kw.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        # Check inputs
        assert self.distribution in ("uniform", "gaussian", "normal")
        assert self.normalization in ("svd", "eig", "multiply")

        # Implement these in the future
        if self.is_symmetric or self.is_antisymmetric:
            raise NotImplementedError(f"RandomMatrix.__init__: symmetry and antisymmetry not implemented")


    def __call__(self):
        A = self.create_matrix()
        return self.normalize(A)


    def create_matrix(self):

        if "distribution" == "uniform":
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

    def __init__(self, **kw):
        super().__init__(**kw)
        assert self.density is not None
        assert self.format is not None


    def create_matrix(self):

        if "distribution" == "uniform":
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

