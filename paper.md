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

ESNs are a conceptually simple Recurrent Neural Network architecture, and
as a result, many scientists who use ESNs implement them from scratch.
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
and the [`cupy-xarray`](https://cupy-xarray.readthedocs.io/index.html) package
to operate on GPUs.

## Parameter Optimization

Although ESNs do not employ backpropagation to train the internal weights of the
input matrix, adjacency matrix, or bias vector,
their behavior and performance is highly sensitive to a set of
5 scalar parameters.
Moreover, the interaction of these parameters is often not straightforward, and
it is therefore advantageous to optimize these parameter values, as is shown by
@platt_systematic_2022 with an extensive set of experimental results.
Additionally, @platt_constraining_2023 showed that adding invariant metrics to
the loss function, such as the Lyapunov exponent spectrum or simply its leading element,
resulted in networks that generalize better to unseen test data.
@smith_temporal_2023 showed similar results in a higher dimensional system using `xesn`'s
parallelized (lazy) architecture, highlighting that constraining the Kinetic Energy
spectrum when developing ESN based emulators of turbulent flows helped to
preserve small scale features.
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
larger than the input and target space, so forecasting large target spaces
quickly becomes intractable with a single reservoir.
To address this, @pathak_model-free_2018 developed a parallelization strategy
so that multiple, semi-independent reservoirs make predictions of a single, high dimensional
system.
This parallelization was generalized for multiple dimensions by
@arcomano_machine_2020 and @smith_temporal_2023, the latter of which serves
as the basis for `xesn`.

`Xesn` enables prediction for multi-dimensional systems by integrating its high
level operations with `xarray` [@hoyer_xarray_2017].
As with `xarray`, users refer to dimensions based on their named axes (e.g., "x",
"latitude", or "time" instead of logical indices 0, 1, 2).
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



# Computational Performance

As discussed in the [Statement of Need](#statement-of-need), one purpose of
`xesn` is to provide a parallelized ESN implementation, which we achieve using
`dask` [@dask_2016].
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

1. Purely threaded mode (CPU only).
2. The relevant default "LocalCluster" (i.e., single node) configuration for our resources.
3. A `LocalCluster` with 1 `dask` worker per group. On GPUs, this assumes 1 GPU per worker
   and we are able to use a maximum of 8 workers due to our available resources.

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


# Acknowledgements

T.A. Smith and S.G. Penny acknowledge support from NOAA Grant NA20OAR4600277.
S.G. Penny and J.A. Platt acknowledge support from the Office of Naval Research (ONR) Grants
N00014-19-1-2522 and N00014-20-1- 2580.
T.A. Smith acknowledges support from
the Cooperative Institute for Research in Environmental Sciences (CIRES) at the
University of Colorado Boulder.

# References
