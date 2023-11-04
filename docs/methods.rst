Methodology Overview
====================

This package implements Echo State Networks (ESNs), which were introduced by
:cite:t:`jaeger_echo_2001`.
ESNs are a Recurrent Neural Network architecture that are in a class of
techniques referred to as Reservoir Computing.
One defining characteristic of these techniques is that all internal weights are
determined by a handful of global or "macro-scale" scalar parameters, thereby avoiding
problems during backpropagation and reducing training time dramatically.

Basic ESN Architecture
######################

The basic ESN architecture that is implemented by the :class:`xesn.ESN` class in this
package is defined as follows: 

.. math::
   \mathbf{r}(n + 1) = (1 - \alpha) \mathbf{r}(n) +
    \alpha \tanh( \mathbf{W}\mathbf{r} + \mathbf{W}_\text{in}\mathbf{u}(n) +
   \mathbf{b})

.. math::
   \hat{\mathbf{v}}(n + 1) = \mathbf{W}_\text{out} \mathbf{r}(n+1)

Here :math:`\mathbf{r}(n)\in\mathbb{R}^{N_r}` is the hidden our reservoir state,
:math:`u(n)` is the input system state, and
:math:`\hat{\mathbf{v}}(n)` is the estimated target or output system state, all at
timestep :math:`n`.
This form is a "leaky" reservoir with the leak rate parameter :math:`\alpha`
determining how much of the previous hidden state to propagate forward in time.

Internal Weights
----------------

The adjacency matrix :math:`\mathbf{W}\in\mathbb{R}^{N_r \times N_r}`,
the input matrix :math:`\mathbf{W}_\text{in}\in\mathbb{R}^{N_r \times N_u}`,
and the bias vector :math:`\mathbf{b}\in\mathbb{R}^{N_r}`
are initialized with random elements, and usually re-scaled.
For instance, it is common to use

.. math::
   \mathbf{W} = \dfrac{\rho}{{\lambda}_\text{max}\left(\hat{\mathbf{W}}\right)}
   \hat{\mathbf{W}}

where the elements of :math:`\hat{\mathbf{W}}` are chosen from a uniform distribution ranging from -1 to 1,
and :math:`{\lambda}_\text{max}` denotes the spectral radius of
:math:`\hat{\mathbf{W}}` such that the factor :math:`\rho` re-scales the
adjacency matrix to attain a particular spectral radius.
The input matrix form is commonly chosen as

.. math::
   \mathbf{W}_\text{in} = \sigma\hat{\mathbf{W}}_\text{in}

where the matrix is usually dense with elements
:math:`w_{i,j}\sim\mathcal{U}[-1,1]`, and scaled simply by the scalar
:math:`\sigma`.
Similarly the bias vector is often chosen as

.. math::
   \mathbf{b} = \sigma_b\hat{\mathbf{b}}

where the factor :math:`\sigma_b` re-scales the randomly initialized vector
:math:`\hat{\mathbf{b}}` with elements :math:`b_i\sim\mathcal{U}[-1,1]`.

However, in xesn the matrix generation is fairly general.
For example a normal or uniform distribution can be used and a variety of
re-scaling or normalization techniques can be used for either of the matrices.
Please see the
:class:`xesn.RandomMatrix` and :class:`xesn.SparseRandomMatrix` classes for all available
options.

Training
--------

The weights in the readout matrix :math:`\mathbf{W}_\text{out}` are learned
during training, which aims to minimize the following loss function

.. math::
   \mathcal{J}(\mathbf{W}_\text{out}) =
    \dfrac{1}{2}\sum_{n=1}^{N_{\text{train}}} ||\mathbf{W}_\text{out}\mathbf{r}(n) -
    \mathbf{v}(n)||_2^2 
    +
    \dfrac{\beta}{2}||\mathbf{W}_\text{out}||_F^2

Here :math:`\mathbf{v}(n)` is the training data at timestep :math:`n`, 
:math:`||\mathbf{A}||_F = \sqrt{Tr(\mathbf{A}\mathbf{A}^T)}` is the Frobenius
norm, :math:`N_{\text{train}}` is the number of timesteps used for training,
and :math:`\beta` is a Tikhonov regularization parameter chosen to improve
numerical stability and prevent overfitting.

Due to the fact that the weights in the adjacency matrix, input matrix, and bias
vector are fixed, the readout matrix weights can be compactly written as the
solution to the linear ridge regression problem

.. math::
   \mathbf{W}_\text{out} = \mathbf{V}\mathbf{R}^T
    \left(\mathbf{R}\mathbf{R}^T + \beta\mathbf{I}\right)^{-1}

where we obtain the solution from `scipy.linalg.solve
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html>`_ 
on CPUs
or `cupy.linalg.solve
<https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.solve.html>`_
on GPUs.
Here :math:`\mathbf{I}` is the identity matrix and
the hidden and target states are expressed in matrix form by concatenating
each time step "column-wise":
:math:`\mathbf{R} = (\mathbf{r}(1) \, \mathbf{r}(2) \, \cdots \, \mathbf{r}(N_{\text{train}}))`
and similarly
:math:`\mathbf{V} = (\mathbf{v}(1) \, \mathbf{v}(2) \, \cdots \, \mathbf{v}(N_{\text{train}}))`.

TODO
----

- note above in matrices that randommatrix and sparserandommatrix are being used
  under the hood by ESN, and so reference to them is just that ... for reference

- summarize the macro parameters?

- mention that only linear readout is supported, given results by Platt et al


Distributed ESN Architecture
############################

It is common to use hidden states that are :math:`\mathcal{O}(10)` to :math:`\mathcal{O}(100)`
times larger than the target system dimension.
In applications that have high dimensional system states, it becomes
necessary to employ a parallelization strategy to distribute the target and
hidden states across many semi-independent networks.
xesn accomplishes this with a generalization of the algorithm introduced by
:cite:t:`pathak_model-free_2018`, where we use
`dask <https://www.dask.org/>`_ to parallelize the
computations.

Example: SQG Turbulence Dataset
-------------------------------

We describe the parallelization strategy based on the dataset used by
:cite:t:`smith_temporal_2023`, which was generated by a model for
Surface Quasi-Geostrophic turbulence.
For the purposes of this discussion, all that matters is the size of the
dataset, which is illustrated below, and more details can be found in Section 2
of :cite:t:`smith_temporal_2023`.


.. image:: images/chunked-sqg.jpg
   :width: 500
   :align: center



The dataset has 3 spatial dimensions :math:`(x, y, z)`, and evolves in time, so
that the shape is :math:`(N_x = 64, N_y = 64, N_z = 2, N_{time})`.
We first subdivide the domain into smaller chunks along the :math:`x` and :math:`y`
dimensions, akin to domain decomposition techniques in General Circulation
Models.
The subdivisions are defined by specifying a chunk size
(:attr:`xesn.LazyESN.esn_chunks`) to the model.
In the case of our example, the chunk size is 

.. code-block:: python

   {"x": 16, "y": 16, "z": 2, "time": -1}

and the chunks are denoted by the black lines across the domain.
Under the hood, :class:`xesn.LazyESN` assigns a local network to each chunk,
where a single dask worker handles all the computations on each chunk.

Communication between chunks is enabled by defining an overlap region,
harnessing dask's flexible `overlap
<https://docs.dask.org/en/latest/generated/dask.array.overlap.overlap.html>`_
function.
The overlap is defined by specifying the size of the overlap in each direction.
For example

.. code-block:: python

   {"x": 1, "y": 1, "z": 0, "time": 0}

defines a single grid cell overlap in :math:`x` and :math:`y`.
An example of a chunk with the additional overlap region is indicated by the
white box in the figure above.
Note that no overlap or chunking is allowed in the :math:`time` dimension, and
that the boundary must be specified for chunks along the edge of the domain -
see :class:`xesn.LazyESN` for details.

Mathematical Definition
-----------------------

The parallelization is achieved by subdividing the domain into :math:`N_g` chunks, and
assigning individual ESNs to each chunk.
That is, we generate the sets
:math:`\{\mathbf{u}_k \subset \mathbf{u} | k = \{1, 2, ..., N_g\} \}`, where
each local input vector :math:`\mathbf{u}_k` includes the overlap region
discussed above. 
The distributed ESN equations are

.. math::
   \mathbf{r}_k(n + 1) = (1 - \alpha) \mathbf{r}_k(n) +
    \alpha \tanh( \mathbf{W}\mathbf{r}_k + \mathbf{W}_\text{in}\mathbf{u}_k(n) +
   \mathbf{b})

.. math::
   \hat{\mathbf{v}}_k(n + 1) = \mathbf{W}_\text{out}^k \mathbf{r}_k(n+1)

Here :math:`\mathbf{r}_k, \, \mathbf{u}_k \, \mathbf{W}_\text{out}^k, \, \hat{\mathbf{v}}_k`
are the hidden state, input state, readout matrix, and estimated output state
associated with the :math:`k^{th}` data chunk.
The local output state :math:`\hat{mathbf{v}}_k` does not include the
overlap region.
Note that the various macro-scale paramaters
:math:`\{\alpha, \rho, \sigma, \sigma_b, \beta\}` are fixed for all chunks.
Therefore the only components that drive unique hidden states on each chunk are
the different input states :math:`\mathbf{u}_k` and the readout matrices
:math:`\mathbf{W}_\text{out}^k`.

More Generally...
-----------------

- The chunks don't need to be even
- The chunk dimensions have to be first, and time dimension last
- Only two chunk dimensions are regularly tested, but more could be added in the
  future
