---
title: 'xesn: Echo state networks powered by xarray and dask'
tags:
  - Python
  - echo state networks
authors:
  - name: Timothy A. Smith
    orcid: 0000-0003-4463-6126
    affiliation: "1, 2"
    corresponding: true
  - name: Stephen G. Penny
    orcid: 0000-0002-5223-8307
    affiliation: "1, 3"
  - name: Jason A. Platt
	orcid: 0000-0001-6579-6546
    affiliation: 4
  - name: Tse-Chun Chen
	orcid: 0000-0001-6300-5659
    affiliation: 5
affiliations:
 - name: Cooperative Institute for Research in Environmental Sciences (CIRES) at the University of Colorado Boulder, Boulder, CO, USA
   index: 1
 - name: Physical Sciences Laboratory (PSL), National Oceanic and Atmospheric Administration (NOAA), Boulder, CO, USA
   index: 2
 - name: Sofar Ocean, San Francisco, CA, USA
   index: 3
 - name: University of California San Diego (UCSD), La Jolla, CA, USA
   index: 4
 - name: Pacific Northwest National Laboratory, Richland, WA, USA
   index: 5
date: 15 December 2023
bibliography: docs/references.bib

---

# TODO

- [ ] add missing citations
- [ ] change parenthetical citations to inline

# Summary

Xesn is a python package that allows scientists to easily design
Echo State Networks (ESNs) for forecasting problems.
ESNs are a Recurrent Neural Network architecture introduced by [@jaeger_echo_2001]
that are part of a class of techniques termed Reservoir Computing.
One defining characteristic of these techniques is that all internal weights are
determined by a handful of global, scalar parameters, thereby avoiding problems
during backpropagation and reducing training time significantly.
Because this architecture is conceptually simple, many scientists implement ESNs
from scratch, leading to questions about computational performance.
Xesn offers a straightforward, standard implementation of ESNs that operates
efficiently on CPU and GPU hardware.
The package leverages optimization tools to automate the parameter selection
process, so that scientists can reduce the time finding a good architecture and
focus on using ESNs for their domain application.
Importantly, the package flexibly handles forecasting tasks for out-of-core,
multi-dimensional datasets,
eliminating the need to write parallel programming code.
Xesn was initially developed to handle the problem of forecasting weather
dynamics, and so it integrates naturally with Python packages that have become
familiar to weather and climate scientists.
However, the software is ultimately general enough to be utilized in other
domains where ESNs have been useful, such as in economics, signal processing,
and biological applications.


# Statement of need

ESNs are a conceptually simple recurrent neural network architecture.
As shown in section ??
they are defined by a simple set of equations, a single hidden layer, and a
handful of parameters.
Because of the architecture's simplicity, many scientists who use ESNs implement
it from scratch.
While this approach can work well for low dimensional problems, the situation
quickly becomes more complicated when:

1. deploying the code on GPUs,
2. interacting with a parameter optimization algorithm in order to tune the
   model, and
3. parallelizing the architecture for higher dimensional applications.

Xesn is designed to address all of these points.
Additionally, while there are some design flexibilities for the ESN
architectures, the overall interface is streamlined, based on the parameter and
design impact study shown by [@platt_systematic_2022].
Users who require a more general Reservoir Computing framework may prefer to use
ReservoirPy CITE (see discussion below).

## GPU Deployment

At its core, xesn uses NumPy CITE and SciPy [@scipy_2020] to perform array based
operations on CPUs, and it uses CuPy [@cupy_learningsys2017] to operate on GPUs.

## Parameter Optimization

Although ESNs do not employ backpropagation to train the internal weights of the
input matrix, adjacency matrix, or bias vector
their behavior and performance is highly sensitive to a set of
5 hyperparameters
(see section XX for math and ).
Moreover, the interaction of these parameters is often not straightforward, and
it is therefore advantageous to optimize these parameter values
[@platt_systematic_2022].
Xesn enables parameter optimization by integrating with the Surrogate Modeling
Toolbox [@bouhlel_scalable_2020], which has a Bayesian Optimization
implementation.

The parameter optimization process is somewhat complex, requiring the user to
specify a variety of hyperparameters (e.g., the optimization bounds for each
parameter and the maximum number of optimization steps to take).
Additionally, for large problems, the process can have heavy compute
requirements and therefore necessitate HPC or cloud resources.
Xesn provides a simple interface so that the user can specify all of the
settings for training, parameter optimization, and testing with a single yaml
file.
By doing so, all parts of the experiment are more easily reproducible and easier to manage
with scheduling systems like SLURM on HPC environments or in the cloud.

## Scaling to Higher Dimensions

It is typical for ESNs to use a hidden layer that is $\mathcal{O}(10-100)$ times
larger than the input and target space.
Forecasting large target spaces (e.g., $>\mathcal{O}(10^6)$ in weather and
climate modeling) quickly becomes intractable with a single reservoir.
To address this, [@pathak_model-free_2018] developed a parallelization strategy
so that multiple reservoirs can make predictions of a single, high dimensional
system.
This parallelization was generalized for multi dimensions in
[@arcomano_machine_2020] and [@smith_temporal_2023], the latter of which served
as the basis for xesn.

Xesn enables prediction for multi dimensional systems by integrating its high
level operations with xarray CITE.
As with xarray, users refer to dimensions based on their named axes (e.g., "x",
"y", or "time" instead of logical indices 0, 1, 2).
Xesn parallelizes the core array based operations by using dask [@dask_2016]
to map them across available resources, which can include a multi threaded
environment on a laptop or single node, or a distributed computing resource
such as traditional on-premises HPC or in the cloud.


## Existing Reservoir Computing Software

It is important to note that there is already an existing software package in
Python for Reservor Computing, called reservoirpy.
To our knowledge, the purpose of this package is distinctly different.
The focus of reservoirpy is to develop a highly generic framework for Reservoir
Computing, for example, allowing one to change the network node type and graph structure
underlying the reservoir, and allowing for delayed connections.
On the other hand, xesn is focused specifically on implementing ESN
architectures that can scale to high dimensional forecasting tasks.
Additionally, while reservoirpy enables hyperparameter grid search capabilities,
xesn enables Bayesian Optimization as noted above.

Finally, we note the code base for CITE Arcomano, which implements ESNs in
Fortran, which can be used for hybrid physics-ML modeling.

# Background

Where to put this stuff?

* Arcomano: ESNs in multidimensional weather forecasting
* Penny: ESN in DA
* Platt: ESNs constrained by LE do really well
* [@smith_temporal_2023]: ESNs constrained by KE Spectrum (or PSD) preserve
  small scale features in turbulent GFD


# The Architectures Implemented


## Standard ESN Architecture

The basic ESN architecture that is implemented by the `xesn.ESN` class
is defined via the discrete timestepping equations:

\begin{equation}
    \begin{aligned}
        \mathbf{r}(n + 1) &= (1 - \alpha) \mathbf{r}(n) +
            \alpha \tanh( \mathbf{W}\mathbf{r} + \mathbf{W}_\text{in}\mathbf{u}(n) +
            \mathbf{b}) \\
        \hat{\mathbf{v}}(n + 1) &= \mathbf{W}_\text{out} \mathbf{r}(n+1) \, .
    \end{aligned}
\end{equation}

Here $\mathbf{r}(n)\in\mathbb{R}^{N_r}$ is the hidden, or reservoir, state,
$u(n)\in\mathbb{R}^{N_\text{in}}$ is the input system state, and
$\hat{\mathbf{v}}(n)\in\mathbb{R}^{N_\text{out}}$ is the estimated target or output system state, all at
timestep $n$.
During training, $\mathbf{u}$ is the input data, but for inference mode the
prediction $\hat{\mathbf{v}} \rightarrow \mathbf{u}$ takes its place.

The input matrix, adjacency matrix, and bias vector are generally defined as
follows:

\begin{equation}
        \mathbf{W} = \dfrac{\rho}{f(\hat{\mathbf{W}})}
            \hat{\mathbf{W}}
\end{equation}
\begin{equation}
        \mathbf{W}_\text{in} = \dfrac{\sigma}{g(\hat{\mathbf{W}}_\text{in})}
            \hat{\mathbf{W}}_\text{in}
\end{equation}
\begin{equation}
        \mathbf{b} = \sigma_b\hat{\mathbf{b}}
\end{equation}

where $\rho$, $\sigma$, and $\sigma_b$ are scaling factors that are chosen or
optimized by the user.
The denominators $f(\hat{\mathbf{W}})$ and $g(\hat{\mathbf{W}}_\text{in})$ are normalization factors,
based on the user's specification (e.g., largest singular value).
The matrices $\hat{\mathbf{W}}$ and $\hat{\mathbf{W}}_\text{in}$ and vector
$\hat{\mathbf{b}}$ are generated randomly, and the user can specify the
underlying distribution used.

We note that this ESN definition assumes a linear readout, and xesn specifically does
not employ more complicated readout operators because @platt_systematic_2022
showed that this does not matter
when the ESN parameters in Section XX are optimized.

## Training

The weights in the readout matrix $\mathbf{W}_\text{out}$ are learned
during training,
which aims to minimize the following loss function

\begin{equation}
    \mathcal{J}(\mathbf{W}_\text{out}) =
        \dfrac{1}{2}\sum_{n=1}^{N_{\text{train}}} ||\mathbf{W}_\text{out}\mathbf{r}(n) -
        \mathbf{v}(n)||_2^2
        +
        \dfrac{\beta}{2}||\mathbf{W}_\text{out}||_F^2 \, .
\end{equation}

Here $\mathbf{v}(n)$ is the training data at timestep $n$,
$||\mathbf{A}||_F = \sqrt{Tr(\mathbf{A}\mathbf{A}^T)}$ is the Frobenius
norm, $N_{\text{train}}$ is the number of timesteps used for training,
and $\beta$ is a Tikhonov regularization parameter chosen to improve
numerical stability and prevent overfitting.

Due to the fact that the weights in the adjacency matrix, input matrix, and bias
vector are fixed, the readout matrix weights can be compactly written as the
solution to the linear ridge regression problem

\begin{equation}
    \mathbf{W}_\text{out} = \mathbf{V}\mathbf{R}^T
        \left(\mathbf{R}\mathbf{R}^T + \beta\mathbf{I}\right)^{-1}
\end{equation}

where we obtain the solution from `scipy.linalg.solve`
on CPUs
or `cupy.linalg.solve`
on GPUs.
Here $\mathbf{I}$ is the identity matrix and
the hidden and target states are expressed in matrix form by concatenating
each time step "column-wise":
$\mathbf{R} = (\mathbf{r}(1) \, \mathbf{r}(2) \, \cdots \, \mathbf{r}(N_{\text{train}}))$
and similarly
$\mathbf{V} = (\mathbf{v}(1) \, \mathbf{v}(2) \, \cdots \, \mathbf{v}(N_{\text{train}}))$.

## Parallel ESN Architecture

The parallelization is achieved by subdividing the domain into $N_g$ chunks, and
assigning individual ESNs to each chunk.
That is, we generate the sets
$\{\mathbf{u}_k \subset \mathbf{u} | k = \{1, 2, ..., N_g\}\}$, and
where each local input vector $\mathbf{u}_k$ includes the overlap region
discussed above.
The distributed ESN equations are

\begin{equation}
    \begin{aligned}
        \mathbf{r}_k(n + 1) &= (1 - \alpha) \mathbf{r}_k(n) +
            \alpha \tanh( \mathbf{W}\mathbf{r}_k + \mathbf{W}_\text{in}\mathbf{u}_k(n) +
            \mathbf{b}) \\
        \hat{\mathbf{v}}_k(n + 1) &= \mathbf{W}_\text{out}^k \mathbf{r}_k(n+1)
    \end{aligned}
\end{equation}

Here $\mathbf{r}_k, \, \mathbf{u}_k \, \mathbf{W}_\text{out}^k, \, \hat{\mathbf{v}}_k$
are the hidden state, input state, readout matrix, and estimated output state
associated with the $k^{th}$ data chunk.
The local output state $\hat{\mathbf{v}}_k$ does not include the
overlap region.
Note that the various macro-scale paramaters
$\{\alpha, \rho, \sigma, \sigma_b, \beta\}$ are fixed for all chunks.
Therefore the only components that drive unique hidden states on each chunk are
the different input states $\mathbf{u}_k$ and the readout matrices
$\mathbf{W}_\text{out}^k$.

## Parameter optimization

* Use smt
* can constrain MSE and PSD as in Smith et al
* can't be on GPUs

# Potential Scaling Results



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](docs/images/chunked-sqg.jpg)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](docs/images/chunked-sqg.jpg){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
