
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
    """Creates a random numpy or cupy based matrix via :meth:`__call__`.
    Use input arguments to control the shape, generator distribution, normalization, and
    the random number generator seed.
    """

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
        """The main routine to use. Creates and normalizes a matrix as specified by attributes.

        Returns:
            A (array_like): numpy or cupy generated random matrix
        """
        A = self.create_matrix()
        return self.normalize(A)


    def create_matrix(self):
        """Create either a uniform or normal random matrix.

        Returns:
            A (array_like): with entries drawn from a standard normal or [-1,1]
                uniform distribution
        """

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
        """Rescale a random matrix either through simple multiplication, or
        by first normalizing by the matrix's spectral radius or induced 2 norm

        Args:
            A (array_like): random matrix

        Returns:
            A (array_like): rescaled matrix, based on class attributes
        """

        denominator = 1.0
        if self.normalization == "svd":
            s = linalg.svd(A, compute_uv=False, full_matrices=False)
            denominator = xp.max(xp.abs(s))

        elif self.normalization == "eig":
            s = linalg.eigvals(A)
            denominator = xp.max(xp.abs(s))


        return self.factor / denominator * A


class SparseRandomMatrix(RandomMatrix):
    """Similar to RandomMatrix, but used to create a sparse random matrix.
    Additional controls are the density and sparse array layout.
    """

    density         = None
    format          = "coo" # scipy's default

    def __init__(self, n_rows, n_cols, distribution, density, **kw):

        super().__init__(
                n_rows=n_rows,
                n_cols=n_cols,
                distribution=distribution,
                **kw)

        self.density    = density


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

