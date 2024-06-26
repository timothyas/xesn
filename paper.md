---
title: 'xesn: Echo state networks powered by xarray and dask'
tags:
  - Python
  - echo state networks
  - xarray
  - dask
  - numpy
  - cupy
authors:
  - name: Timothy A. Smith
    orcid: 0000-0003-4463-6126
    affiliation: "1"
    corresponding: true
  - name: Stephen G. Penny
    orcid: 0000-0002-5223-8307
    affiliation: "2, 3"
  - name: Jason A. Platt
    orcid: 0000-0001-6579-6546
    affiliation: 4
  - name: Tse-Chun Chen
    orcid: 0000-0001-6300-5659
    affiliation: 5
affiliations:
 - name: Physical Sciences Laboratory (PSL), National Oceanic and Atmospheric Administration (NOAA), Boulder, CO, USA
   index: 1
 - name: Sofar Ocean, San Francisco, CA, USA
   index: 2
 - name: Cooperative Institute for Research in Environmental Sciences (CIRES) at the University of Colorado Boulder, Boulder, CO, USA
   index: 3
 - name: University of California San Diego (UCSD), La Jolla, CA, USA
   index: 4
 - name: Pacific Northwest National Laboratory, Richland, WA, USA
   index: 5
date: 15 December 2023
bibliography: docs/references.bib

---

# Summary

`Xesn` is a python package that allows scientists to easily design
Echo State Networks (ESNs) for forecasting problems.
ESNs are a Recurrent Neural Network architecture introduced by @jaeger_echo_2001
that are part of a class of techniques termed Reservoir Computing.
One defining characteristic of these techniques is that all internal weights are
determined by a handful of global, scalar parameters, thereby avoiding problems
during backpropagation and reducing training time significantly.
Because this architecture is conceptually simple, many scientists implement ESNs
from scratch, leading to questions about computational performance.
`Xesn` offers a straightforward, standard implementation of ESNs that operates
efficiently on CPU and GPU hardware.
The package leverages optimization tools to automate the parameter selection
process, so that scientists can reduce the time finding a good architecture and
focus on using ESNs for their domain application.
Importantly, the package flexibly handles forecasting tasks for out-of-core,
multi-dimensional datasets,
eliminating the need to write parallel programming code.
`Xesn` was initially developed to handle the problem of forecasting weather
dynamics, and so it integrates naturally with Python packages that have become
familiar to weather and climate scientists such as `xarray` [@hoyer_xarray_2017].
However, the software is ultimately general enough to be utilized in other
domains where ESNs have been useful, such as in
signal processing [@jaeger_harnessing_2004].


# Statement of Need

ESNs are a conceptually simple Recurrent Neural Network architecture.
As shown in the [Implemented Architectures](#implemented-architectures) section,
they are defined by a simple set of equations, a single hidden layer, and a
handful of parameters.
Because of the architecture's simplicity, many scientists who use ESNs implement
them from scratch.
While this approach can work well for low dimensional problems, the situation
quickly becomes more complicated when:

1. deploying the code on GPUs,
2. interacting with a parameter optimization algorithm in order to tune the
   model, and
3. parallelizing the architecture for higher dimensional applications.

`Xesn` is designed to address all of these points.
Additionally, while there are some design flexibilities for the ESN
architectures, the overall interface is streamlined, based on the parameter and
design impact study shown by @platt_systematic_2022.

## GPU Deployment

At its core, `xesn` uses NumPy [@harris_array_2020] and SciPy [@scipy_2020] to perform array based
operations on CPUs.
The package then harnesses the CPU/GPU agnostic code capabilities afforded by CuPy [@cupy_learningsys2017]
to operate on GPUs.

## Parameter Optimization

Although ESNs do not employ backpropagation to train the internal weights of the
input matrix, adjacency matrix, or bias vector,
their behavior and performance is highly sensitive to a set of
5 scalar parameters
(see [Macro-Scale Parameters](#macro-scale-parameters)).
Moreover, the interaction of these parameters is often not straightforward, and
it is therefore advantageous to optimize these parameter values, as is shown by
@platt_systematic_2022 with an extensive set of experimental results.
Additionally, @platt_constraining_2023 showed that adding invariant metrics to
the loss function, such as the Lyapunov exponent spectrum or simply its leading element,
resulted in networks that generalize better to unseen test data.
@smith_temporal_2023 showed similar results, highlighting that constraining the Kinetic Energy
spectrum when developing ESN based emulators of Surface Quasi-Geostrophic
Turbulence helped to preserve small scale features in the flow.
As a somewhat generic and efficient implementation of these metrics,
`xesn` offers the capability to constrain the
system's Power Spectral Density during parameter optimization, in addition to a
more traditional
Normalized Root Mean Squared Error loss function.

`Xesn` enables parameter optimization by integrating with the Surrogate Modeling
Toolbox [@bouhlel_scalable_2020], which has a Bayesian Optimization
implementation.
The parameter optimization process is in general somewhat complex, requiring the user to
specify a variety of hyperparameters (e.g., the optimization bounds for each
parameter and the maximum number of optimization steps to take).
Additionally, for large problems, the process can have heavy compute
requirements and therefore necessitate HPC or cloud resources.
`Xesn` provides a simple interface so that the user can specify all of the
settings for training, parameter optimization, and testing with a single YAML
file.
By doing so, all parts of the experiment are more easily reproducible and easier to manage
with scheduling systems like SLURM on HPC environments or in the cloud.

## Scaling to Higher Dimensions

It is typical for ESNs to use a hidden layer that is $\mathcal{O}(10-100)$ times
larger than the input and target space.
Forecasting large target spaces (e.g., $>\mathcal{O}(10^6)$ in weather and
climate modeling) quickly becomes intractable with a single reservoir.
To address this, @pathak_model-free_2018 developed a parallelization strategy
so that multiple, semi-independent reservoirs make predictions of a single, high dimensional
system.
This parallelization was generalized for multiple dimensions by
@arcomano_machine_2020 and @smith_temporal_2023, the latter of which serves
as the basis for `xesn`.

`Xesn` enables prediction for multi-dimensional systems by integrating its high
level operations with `xarray` [@hoyer_xarray_2017].
As with `xarray`, users refer to dimensions based on their named axes (e.g., "x",
"y", or "time" instead of logical indices 0, 1, 2).
`Xesn` parallelizes the core array based operations by using `dask` [@dask_2016]
to map them across available resources, which can include a multi-threaded
environment on a laptop or single node, or a distributed computing resource
such as traditional on-premises HPC or in the cloud.


## Existing Reservoir Computing Software

It is important to note that there is already an existing software package in
Python for Reservoir Computing, called `ReservoirPy` [@Trouvain2020].
To our knowledge, the purpose of this package is distinctly different.
The focus of `ReservoirPy` is to develop a highly generic framework for Reservoir
Computing, for example, allowing one to change the network node type and graph structure
underlying the reservoir, and allowing for delayed connections.
On the other hand, `xesn` is focused specifically on implementing ESN
architectures that can scale to multi-dimensional forecasting tasks.
Additionally, while `ReservoirPy` enables hyperparameter grid search capabilities
via Hyperopt [@hyperopt], `xesn` enables Bayesian Optimization as noted above.

Finally, we note the code base used by [@arcomano_machine_2020;@arcomano_hybrid_2022;@arcomano_hybrid_2023],
available at [@arcomano_code].
The code implements ESNs in Fortran, and focuses on using ESNs for hybrid physics-ML modeling.

# Implemented Architectures


## Standard ESN Architecture

The ESN architecture that is implemented by the `xesn.ESN` class
is defined by the discrete timestepping equations:

\begin{equation}
    \begin{aligned}
        \mathbf{r}(n + 1) &= (1 - \alpha) \mathbf{r}(n) +
            \alpha \tanh( \mathbf{W}\mathbf{r}(n) + \mathbf{W}_\text{in}\mathbf{u}(n) +
            \mathbf{b}) \\
        \hat{\mathbf{v}}(n + 1) &= \mathbf{W}_\text{out} \mathbf{r}(n+1) \, .
    \end{aligned}
    \label{eq:esn}
\end{equation}

Here $\mathbf{r}(n)\in\mathbb{R}^{N_r}$ is the hidden, or reservoir, state,
$u(n)\in\mathbb{R}^{N_\text{in}}$ is the input system state, and
$\hat{\mathbf{v}}(n)\in\mathbb{R}^{N_\text{out}}$ is the estimated target or output system state, all at
timestep $n$.
During training, $\mathbf{u}$ is the input data, but for inference mode the
prediction $\hat{\mathbf{v}} \rightarrow \mathbf{u}$ takes its place.
This ESN implementation is *eager*, in the sense that all of the
inputs, network parameters, targets (during training), and predictions are held
in memory.

The input matrix, adjacency matrix, and bias vector are generally defined as
follows:

\begin{equation}
    \begin{aligned}
        \mathbf{W} &= \dfrac{\rho}{f(\hat{\mathbf{W}})}
            \hat{\mathbf{W}} \\
        \mathbf{W}_\text{in} &= \dfrac{\sigma}{g(\hat{\mathbf{W}}_\text{in})}
            \hat{\mathbf{W}}_\text{in} \\
        \mathbf{b} &= \sigma_b\hat{\mathbf{b}}
    \end{aligned}
    \label{eq:random}
\end{equation}

where $\rho$, $\sigma$, and $\sigma_b$ are scaling factors that are chosen or
optimized by the user.
The denominators $f(\hat{\mathbf{W}})$ and $g(\hat{\mathbf{W}}_\text{in})$ are normalization factors,
based on the user's specification (e.g., largest singular value).
The matrices $\hat{\mathbf{W}}$ and $\hat{\mathbf{W}}_\text{in}$ and vector
$\hat{\mathbf{b}}$ are generated randomly, and the user can specify the
underlying distribution used.

We note that this ESN definition assumes a linear readout, and `xesn` specifically does
not employ more complicated readout operators because @platt_systematic_2022
showed that this does not matter
when the ESN parameters listed in
[Macro-Scale Parameters](#macro-scale-parameters) are optimized.

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
    \label{eq:loss}
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
    \label{eq:Wout}
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

## Macro-Scale Parameters

Some of the most important macro-scale parameters aside from the
size of the reservoir that must be specified are:

- input matrix scaling, $\sigma$
- adjacency matrix scaling, $\rho$
- bias vector scaling, $\sigma_b$
- Tikhonov parameter, $\beta$
- leak rate, $\alpha$

As noted in the [Parameter Optimization](#parameter-optimization) section,
these can be optimized via the Bayesian Optimization algorithm in the Surrogate
Modeling Toolbox [@bouhlel_scalable_2020]
by minimizing a combination of Normalization Root Mean Squared Error and Power
Spectral Density Error terms.

## Parallel ESN Architecture

We describe the parallelization strategy with an explicit example, using the dataset
from @smith_temporal_2023, which was generated by
a Surface Quasi-Geostrophic Turbulence model [written by Jeff Whitaker](https://github.com/jswhit/sqgturb),
following [@blumen_uniform_1978; @held_surface_1995].
For the purposes of this discussion, all that matters is the size of the
dataset, which is illustrated below, and more details can be found in Section 2
of @smith_temporal_2023.


![A snapshot of potential temperature from
[a model for Surface Quasi-Geostrophic Turbulence](https://github.com/jswhit/sqgturb)
[@blumen_uniform_1978; @held_surface_1995],
with dimensions labelled in order to illustrate the ESN parallelization scheme.
\label{fig:sqg}
](docs/images/chunked-sqg.jpg){width=50%}

The dataset has 3 spatial dimensions $(x, y, z)$, and evolves in time, so
that the shape is $(N_x = 64, N_y = 64, N_z = 2, N_{time})$.
Parallelization is achieved in this case by subdividing the domain into smaller groups
along the $x$ and $y$ dimensions, akin to domain decomposition
techniques in General Circulation Models.
In the case of our example, each group (or chunk) is defined with size
```python
esn_chunks={"x": 16, "y": 16, "z": 2}
```
and these groups are denoted by the black lines across the domain in
\autoref{fig:sqg}.
Under the hood, `xesn.LazyESN` assigns a local network to each chunk,
where each chunk becomes a separate `dask` task.
Note that unlike `xesn.ESN`, `xesn.LazyESN` does not load all data into memory,
but instead lazily operates on the data via the `dask.Array` API.

Communication between groups is enabled by defining an overlap or halo region,
harnessing `dask`'s flexible overlap function.
In our example, the overlap region is shown for a single group by the white box,
and for instance is defined with size:
```python
overlap={"x": 1, "y": 1, "z": 0}
```
that is, there is a single grid cell overlap in $x$ and $y$, but no overlap in
the vertical.

The ESN parallelization is achieved by assigning individual ESNs to each group.
That is, we generate the sets
$\{\mathbf{u}_k \subset \mathbf{u} | k = \{1, 2, ..., N_g\}\}$,
where each local input vector $\mathbf{u}_k$ includes the overlap region
discussed above.
The distributed ESN equations are

\begin{equation}
    \begin{aligned}
        \mathbf{r}_k(n + 1) &= (1 - \alpha) \mathbf{r}_k(n) +
            \alpha \tanh( \mathbf{W}\mathbf{r}_k(n) + \mathbf{W}_\text{in}\mathbf{u}_k(n) +
            \mathbf{b}) \\
        \hat{\mathbf{v}}_k(n + 1) &= \mathbf{W}_\text{out}^k \mathbf{r}_k(n+1)
    \end{aligned}
    \label{eq:lazyesn}
\end{equation}

Here $\mathbf{r}_k, \, \mathbf{u}_k \, \mathbf{W}_\text{out}^k, \, \hat{\mathbf{v}}_k$
are the hidden state, input state, readout matrix, and estimated output state
associated with the $k^{th}$ group.
Note that the local output state $\hat{\mathbf{v}}_k$ does not include the
overlap region.
Currently, the various macro-scale paramaters
$\{\alpha, \rho, \sigma, \sigma_b, \beta\}$ are fixed for all groups.
Therefore, the only components that drive unique hidden states on each chunk are
the different input states $\mathbf{u}_k$ and the readout matrices
$\mathbf{W}_\text{out}^k$.
Additionally, because the solution to \autoref{eq:loss} is linear, the readout matrices are simply
\begin{equation}
    \mathbf{W}^k_\text{out} = \mathbf{V}_k\mathbf{R}_k^T
        \left(\mathbf{R}_k\mathbf{R}_k^T + \beta\mathbf{I}\right)^{-1} \, ,
    \label{eq:WoutLazy}
\end{equation}
where each readout matrix can be computed independently of one another.


# Computational Performance

As discussed in the [Statement of Need](#statement-of-need), one purpose of
`xesn` is to provide a parallelized ESN implementation, which we achieve using
`dask` [@dask_2016].
One advantage of `dask` is that it provides a highly flexible, task-based
parallelization framework such that the same code can be parallelized using a
combination of threads and processes, and deployed in a variety of settings from
a laptop to HPC platforms, either on premises or in the cloud.
Here we show brief scaling results in order to give some practical guidance on
how to best configure `dask` when using the parallelized `LazyESN` architecture.

## Standard (Eager) ESN Performance

![Walltime and memory usage for the standard ESN architecture for two different
system sizes ($N_u$) and a variety of reservoir sizes ($N_r$).
Walltime is captured with Python's `time` module, and memory is captured with
[memory-profiler](https://pypi.org/project/memory-profiler/)
for the CPU runs and with
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
for the GPU runs.
The dotted lines indicate theoretical scaling of memory, where
$a=250,000$ and $b=20,000$ are empirically derived constants, and
$c=8\cdot10^9$ is a conversion to GB.
\label{fig:eager}
](scaling/eager-scaling.pdf){ width=100% }

For reference, in \autoref{fig:eager} we show the walltime and memory usage involved for
training the
standard (eager) `ESN` architecture as a function of the input dimension $N_u$ and
reservoir size $N_r$.
Each data point in \autoref{fig:eager} involved running the following commands:
```python
from xesn import Driver
driver = Driver(config="scaling/config-eager.yaml")
driver.overwrite_config(
    {"esn": {
        "n_input": N_u,
        "n_output": N_u,
        "n_reservoir": N_r,
    }
})
driver.run_training()
```
We ran the scaling tests in the `us-central-1c` zone on Google Cloud Platform, using
a single `c2-standard-60` instance to test the CPU (NumPy) implementation
and a single `a2-highgpu-8g` (i.e., with 8 A100 cards) instance to test the GPU
(CuPy) implementation.
The training data was generated from the Lorenz96 model
[@lorenz_predictability_1996] with dimensions
$N_u=\{16,256\}$,
and we generated 80,000 total samples in the training dataset.

In the CPU tests, walltime scales quadratically with the reservoir size, while
it is mostly constant on a GPU.
For this problem, it becomes advantageous to use GPUs once the reservoir size is
approximately $N_r=8,000$ or greater.
Notably, we achieve a speedup factor of 2.5-3
for the large reservoir case ($N_r=16,000$), when comparing GPU to CPU walltime.
In both the CPU and GPU tests, memory scales quadratically with reservoir size,
although the increasing memory usage with reservoir size is more dramatic on the
CPU than GPU.
This result serves as a motivation for our parallelized architecture.

## Parallel (Lazy) Architecture Strong Scaling Results

In order to evaluate the performance of the parallelized architecture, we take
the Lorenz96 system with dimension $N_u=256$ and subdivide the domain into
$N_g = \{2, 4, 8, 16, 32\}$ groups.
We then fix the reservoir size so that $N_r*N_g = 16,000$, so that the problem
size is more or less fixed and the timing results reflect strong scaling.
The training task and resources used are otherwise the same as for the standard
ESN results shown in \autoref{fig:eager}.
We then create 3 different `dask.distributed` Clusters, testing:

1. Purely threaded mode.
   ```python
   # CPUs only
   from distributed import Client
   client = Client(processes=False)
   ```

2. The relevant default "LocalCluster" (i.e., single node) configuration for our resources.
   ```python
   # On CPU
   from distributed import Client
   client = Client()

   # On GPU
   from dask_cuda import LocalCUDACluster
   cluster = LocalCUDACluster()
   client = Client(cluster)
   ```

3. A `LocalCluster` with 1 `dask` worker per group. On GPUs, this assumes 1 GPU per worker
   and we are able to use a maximum of 8 workers due to our available resources.
   ```python
   # On CPUs
   from distributed import Client
   client = Client(n_workers = n_g)

   # On GPUs
   from dask_cuda import LocalCUDACluster
   cluster = Cluster(CUDA_VISIBLE_DEVICES="0,1") # e.g. for N_groups=2
   client = Client(cluster)
   ```

There are, of course, many more ways to configure a `dask` cluster, but these
three examples should provide some guidance for even larger problems that require
e.g.,
[`dask-jobqueue`](https://jobqueue.dask.org/en/latest/)
or
[`dask-cloudprovider`](https://cloudprovider.dask.org/en/latest/).

![Strong scaling results, showing speedup as a ratio of serial training time to
parallel training time as a function of number of groups or subdomains of the
Lorenz96 system.
Serial training time is evaluated with $N_u=256$ and $N_r=16,000$ with
`xesn.ESN` from \autoref{fig:eager}, and parallel training time uses `xesn.LazyESN` with
the number of groups as shown.
See text for a description of the different schedulers used.
\label{fig:lazy}
](scaling/lazy-scaling.pdf){ width=100% }

\autoref{fig:lazy} shows the strong scaling results of `xesn.LazyESN` for each of these
cluster configurations, where each point shows the ratio of the
walltime with the standard (serial) architecture to the lazy (parallel)
architecture with $N_g$ groups.
On CPUs, using 1 `dask` worker process per ESN group generally scales well,
which makes sense because each group is trained entirely independently.
However, the two exceptions to this rule of thumb are as follows.

1. When there are only 2 groups, the threaded scheduler does slightly better,
   presumably because of the lack of overhead involved with multiprocessing.

2. When $N_g$ is close to the default provided by `dask`, it might be best to
   use that default.

On GPUs, the timing is largely determined by how many workers (GPUs) there are
relative to the number of groups.
When the number of workers is less than the number of groups, performance is
detrimental.
However, when there is at least one worker per group, the timing is almost the
same as for the single worker case, only improving performance by 10-20%.
While the strong scaling is somewhat muted, the invariance of walltime to
reservoir size in \autoref{fig:eager} and number of groups in
\autoref{fig:lazy} means that the distributed GPU
implementation is able to tackle larger problems at roughly the same
computational cost.



# Conclusions

We have presented `xesn`, a Python package that allows scientists to implement ESNs
for a variety of forecasting problems.
The package relies on a software stack that is already familiar to weather and
climate scientists, and allows users to
(1) easily deploy on GPUs,
(2) use Bayesian Optimization to easily design skillful networks, and
(3) scale the workflow, including Bayesian Optimization, to high dimensional, multivariate problems.
We have additionally provided performance results in order to help scientists
scale their networks to large problems.
The main current limitation of `xesn` is that it does not enable Bayesian
Optimization on GPUs, due to the Surrogate Modeling Toolbox's current lack of GPU
integration.
Future versions could address this limitation by implementing Bayesian Optimization in the source code, or
integrate with `Ray Tune` [@liaw2018tune].
Additionally, future versions could incorporate JAX [@jax2018github] in order to
speed up the CPU implementation and deploy on GPU or TPU hardware.


# Acknowledgements

T.A. Smith and S.G. Penny acknowledge support from NOAA Grant NA20OAR4600277.
S.G. Penny and J.A. Platt acknowledge support from the Office of Naval Research (ONR) Grants
N00014-19-1-2522 and N00014-20-1- 2580.
T.A. Smith acknowledges support from
the Cooperative Institute for Research in Environmental Sciences (CIRES) at the
University of Colorado Boulder.

# References
