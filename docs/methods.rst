Methodology Overview
====================

This package implements Echo State Networks (ESNs), which were introduced by
[Jaeger_2001]_.
ESNs are a Recurrent Neural Network architecture that are in a class of
techniques referred to as Reservoir Computing.
One defining characteristic of these techniques is that all internal weights are
determined by a handful of global or "macro-scale" scalar parameters, thereby avoiding
problems during backpropagation and reducing training time dramatically.

Basic ESN Architecture
######################

The basic ESN architecture that is implemented by the :class:`ESN` class in this
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
:class:`RandomMatrix` and :class:`SparseRandomMatrix` classes for all available
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
:math:`||\mathbf{A}|| = \sqrt{Tr(\mathbf{A}\mathbf{A}^T)}` is the Frobenius
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

Parameters
----------

TODO: summarize the macro parameters?


Distributed ESN Architecture
############################

It is common to use hidden states that are :math:`\mathcal{O}(10)` to :math:`\mathcal{O}(100)`
times larger than the target system dimension.
In applications that have high dimensional system states, it is therefore
necessary to employ a parallelization strategy to distribute the target and
hidden states across many semi-independent networks.
xesn accomplishes this with a generalization of the algorithm introduced by
[Pathak_et_al_2018]_, using `dask <https://www.dask.org/>`_ to parallelize the
computations.

The system domain is subdivided into :math:`N_g` groups based on their location, 

