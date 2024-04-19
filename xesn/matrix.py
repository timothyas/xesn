
from scipy import stats
from scipy.sparse.linalg import svds as scipy_svds

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

    Args:
        n_rows (int): number of rows in the matrix
        n_cols (int): number of columns in the matrix
        distribution (str): distribution to draw elements of the matrix from, either "uniform", or "gaussian" and "normal" are recognized
        normalization (str, optional): method used to rescale the matrix, see :meth:`normalize`
        factor (float, optional): factor to rescale the matrix with after it's been normalized
        random_seed (int, optional): used to control the RNG for matrix generation

    Example:
        Create a 2x2 random matrix with entries from a uniform distribution ranging from -5 to 5

        >>> from xesn import RandomMatrix
        >>> mat = RandomMatrix(2, 2, factor=5, distribution="uniform", random_seed=0)
        >>> A = mat()
        >>> A
        array([[0.48813504, 2.15189366],
               [1.02763376, 0.44883183]])

        Note that every time the object is called, a new matrix is generated

        >>> B = mat()
        >>> B
        array([[-0.76345201,  1.45894113],
               [-0.62412789,  3.91773001]])

    Example:
        Create a 2x2 random matrix with entries from a Gaussian distribution with its spectral radius set to 0.9

        >>> from xesn import RandomMatrix ; from scipy.linalg import eigvals
        >>> mat = RandomMatrix(2, 2, factor=0.9, distribution="gaussian", normalization="eig", random_seed=0)

        or equivalently...

        >>> mat = RandomMatrix(2, 2, factor=0.9, distribution="normal", normalization="eig", random_seed=0)
        >>> A = mat()
        >>> A
        array([[0.59414168, 0.13477495],
               [0.32964386, 0.75474407]])
        >>> max(abs(eigvals(A)))
        0.9000000000000001

    Example:
        Create a 2x2 random matrix with entries from a Gaussian distribution with its induced 2-norm set to 1.0

        >>> from xesn import RandomMatrix ; from scipy.linalg import svd
        >>> mat = RandomMatrix(2, 2, factor=1.0, distribution="gaussian", normalization="svd", random_seed=0)
        >>> A = mat()
        >>> A
        array([[0.64082821, 0.14536532],
               [0.35554665, 0.81405043]])
        >>> max(abs(svd(A, compute_uv=False, full_matrices=False)))
        1.0

    """

    __slots__ = (
        "n_rows", "n_cols",
        "distribution", "normalization", "factor",
        "random_seed", "random_state",
    )

    def __init__(
            self,
            n_rows,
            n_cols,
            factor,
            distribution,
            normalization="multiply",
            random_seed=None):

        self.n_rows         = n_rows
        self.n_cols         = n_cols
        self.distribution   = distribution
        self.normalization  = normalization
        self.factor         = factor
        self.random_seed    = random_seed

        # Check inputs
        try:
            assert self.distribution in ("uniform", "gaussian", "normal")
        except AssertionError:
            raise NotImplementedError(f"RandomMatrix.__init__: '{self.distribution}' not recgonized, only 'uniform', 'gaussian'/'normal' are implemented")

        try:
            assert self.normalization in ("svd", "eig", "multiply")
        except AssertionError:
            raise NotImplementedError(f"RandomMatrix.__init__: '{self.normalization}' not recgonized, only 'svd', 'eig', and 'multiply' are implemented")

        if _use_cupy and self.normalization == "eig":
            raise NotImplementedError(f"RandomMatrix.__init__: '{self.normalization}' not available with CuPy.")
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
        by first normalizing by the matrix's spectral radius or induced 2 norm.
        Given the matrix :math:`A` with eigenvalues
        :math:`\{ {\lambda}_1, {\lambda}_2, ..., {\lambda}_n\}`,
        and singular values
        :math:`\{ {\sigma}_1, {\sigma}_2, ..., {\sigma}_n\}`,
        the normalization options return the following:

        normaliation="multiply":

        .. math::
            factor * A

        normalization="eig":

        .. math::
            \dfrac{factor}{\\rho(A)} A

        where :math:`{ \\rho(A) = \max \{ | \lambda_1 | , | \lambda_2 |, ..., | \lambda_n | \} }` is the spectral radius of :math:`A`.

        normalization="svd":

        .. math::
            \dfrac{factor}{\sigma(A)} A

        where :math:`\sigma(A) = \max \{\sigma_1, \sigma_2, ..., \sigma_n\}` is the largest singular value, a.k.a. the induced 2-norm of :math:`A`.


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
    """Similar to :class:`RandomMatrix`, but used to create a sparse random matrix.
    Additional controls are the density and sparse array layout.

    Example:
        Create a sparse 100x100 random matrix with entries from a Gaussian distribution with its spectral radius set to 0.9

        >>> from xesn import SparseRandomMatrix ; from scipy.sparse.linalg import eigs
        >>> mat = SparseRandomMatrix(100, 100, factor=0.9, distribution="normal", normalization="eig", connectedness=1, random_seed=0)
        >>> A = mat()
        >>> A
        <100x100 sparse matrix of type '<class 'numpy.float64'>'
            with 100 stored elements in COOrdinate format>
        >>> A.nnz
        100
        >>> mat.density
        0.01
        >>> max(abs(eigs(A, k=1, which="LM", return_eigenvectors=False)))
        0.9000000000000011

    Args:
        n_rows (int): number of rows in the matrix
        n_cols (int): number of columns in the matrix
        factor (float, optional): factor to rescale the matrix with after it's been normalized
        distribution (str): distribution to draw elements of the matrix from, either "uniform", or "gaussian" and "normal" are recognized
        normalization (str, optional): method used to rescale the matrix, see :meth:`normalize`
        density, sparsity, connectedness (float): use 'density', 'sparsity' or 'connectedness' to specify the number of nonzero values in the matrix, where 'density' sets the fraction of nonzero values, ``sparsity=1 - density``, and ``density = connectedness / n_rows``, but note that 'connectedness' only makes sense for square matrices
        random_seed (int, optional): used to control the RNG for matrix generation
    """

    __slots__ = (
        "density", "format",
    )

    def __init__(
            self,
            n_rows,
            n_cols,
            factor,
            distribution,
            normalization="multiply",
            density=None,
            sparsity=None,
            connectedness=None,
            format="coo",
            random_seed=None):

        super().__init__(
                n_rows=n_rows,
                n_cols=n_cols,
                factor=factor,
                distribution=distribution,
                normalization=normalization,
                random_seed=random_seed)

        self.format = format
        try:
            assert (density is not None) or (sparsity is not None) or (connectedness is not None)
        except AssertionError:
            raise TypeError(f"SparseRandomMatrix.__init__: specify matrix density by one of the following arguments: 'sparsity', 'density', or 'connectedness'")

        if density is not None:
            self.density = density
            try:
                assert (sparsity is None) and (connectedness is None)
            except AssertionError:
                raise TypeError(f"SparseRandomMatrix.__init__: density set to {density}, cannot also specify 'sparsity' and/or 'connectedness'")
        elif sparsity is not None:
            self.density = 1. - sparsity
            try:
                assert (density is None) and (connectedness is None)
            except AssertionError:
                raise TypeError(f"SparseRandomMatrix.__init__: sparsity set to {sparsity}, cannot also specify 'density' and/or 'connectedness'")
        else: # use connectedness
            try:
                assert n_rows == n_cols
            except AssertionError:
                raise TypeError(f"SparseRandomMatrix.__init__: matrix connectivity does not make sense for non-square matrix, use 'density' or 'sparsity' instead")
            self.density = connectedness / n_rows


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
            # cupy.sparse.linalge.svds and cupy.sparse.linalg.eigsh fail when k is close to
            # the shape of A, and seems like when min(A.shape) == k+2
            # see here https://github.com/cupy/cupy/issues/6863
            # so for 3x3 systems compute with numpy
            if _use_cupy and min(A.shape) == 3:
                s = scipy_svds(A.get(), k=1, which="LM", return_singular_vectors=False)
                s = xp.float64(s[0])
            else:
                s = sparse.linalg.svds(A, k=1, which="LM", return_singular_vectors=False)
            denominator = xp.max(xp.abs(s))

        elif self.normalization == "eig":
            s = sparse.linalg.eigs(A, k=1, which="LM", return_eigenvectors=False)
            denominator = xp.max(xp.abs(s))

        A.data = self.factor / denominator * A.data
        return A
