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
date: 1 November 2024
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

ESNs are a conceptually simple Recurrent Neural Network architecture,
leading many scientists who use ESNs to implement them from scratch.
While this approach can work well for low dimensional problems, the situation
quickly becomes more complicated when:

1. deploying the code on GPUs,
2. interacting with a parameter optimization algorithm in order to tune the
   model, and
3. parallelizing the architecture for higher dimensional applications.

`Xesn` is designed to address all of these points.
Additionally, while there are some design flexibilities for the ESN
architectures, the overall interface is streamlined based on the parameter and
design impact study shown by @platt_systematic_2022.

## GPU Deployment

At its core, `xesn` uses NumPy [@harris_array_2020] and SciPy [@scipy_2020] to perform array based
operations on CPUs.
The package then harnesses the CPU/GPU agnostic code capabilities afforded by CuPy [@cupy_learningsys2017]
to operate on GPUs.

## Parameter Optimization

Although ESNs do not employ backpropagation to train internal weights,
their behavior and performance is highly sensitive to a set of
5 scalar parameters.
Moreover, the interaction of these parameters is often not straightforward, and
it is therefore advantageous to optimize these parameter values
[@platt_systematic_2022].
Additionally, @platt_constraining_2023 and @smith_temporal_2023 showed that
adding invariant metrics to the loss function, like the leading Lyapunov
exponent or the Kinetic Energy spectrum, improved generalizability.
As a generic implementation of these metrics,
`xesn` offers the capability to constrain the
system's Power Spectral Density during parameter optimization in addition to a
more traditional mean squared error loss function.

`Xesn` enables parameter optimization by integrating with the Surrogate Modeling
Toolbox [@bouhlel_scalable_2020], which has a Bayesian optimization
implementation.
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
As with `xarray`, users refer to dimensions based on their named axes.
`Xesn` parallelizes the core array based operations by using `dask`
[@dask_2016; @rocklin_scipy_2015]
to map them across available resources, from a laptop to a distributed HPC or
cloud cluster.


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
via Hyperopt [@hyperopt], `xesn` enables Bayesian optimization as noted above.

Another ESN implementation is that of [@arcomano_machine_2020;@arcomano_hybrid_2022;@arcomano_hybrid_2023],
available at [@arcomano_code].
The code implements ESNs in Fortran, and focuses on using ESNs for hybrid physics-ML modeling.


# Computational Performance

Here we show brief scaling results in order to show
how the standard (eager)
[`xesn.ESN`](https://xesn.readthedocs.io/en/latest/generated/xesn.ESN.html#xesn.ESN)
scales with increasing hidden and input dimensions.
Additionally, we provide some baseline results to serve as guidance when
configuring `dask` to use the parallelized
[`xesn.LazyESN`](https://xesn.readthedocs.io/en/latest/generated/xesn.LazyESN.html) architecture.
The scripts used to setup, execute, and visualize these scaling tests can be
found
[here](https://github.com/timothyas/xesn/tree/1524713149efa38a0fd52ecdeb32ca5aacb62693/scaling).
For methodological details on these two architectures, please refer to
[the methods section of the documentation](https://xesn.readthedocs.io/en/latest/methods.html).

## Standard (Eager) ESN Performance

![Wall time and peak memory usage for the standard ESN architecture for two different
system sizes ($N_u$) and a variety of reservoir sizes ($N_r$).
Wall time is captured with Python's `time` module, and peak memory usage is captured with
[memory-profiler](https://pypi.org/project/memory-profiler/)
for the CPU runs and with
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
for the GPU runs.
Note that the peak memory usage for the GPU runs indicates GPU memory usage
only, since this is a typical bottleneck.
The gray and black lines indicate the general trend in
memory usage during the CPU and GPU simulations, respectively.
The empirically derived gray and black curves are a function of the problem size, and
are provided so users can estimate how much memory might be
required for their applications.
The constants are as follows:
$a=250,000$ is ~3 times the total number of samples used,
$b=20,000$ is the batch size, and
$c=8\cdot10^9$ is a conversion to GB.
\label{fig:eager}
](scaling/eager-scaling.pdf){ width=100% }

For reference, in \autoref{fig:eager} we show the wall time and peak memory usage required to
train the
standard (eager) `ESN` architecture as a function of the input dimension $N_u$ and
reservoir size $N_r$.
We ran the scaling tests in the `us-central-1c` zone on Google Cloud Platform (GCP), using
a single `c2-standard-60` instance to test the CPU (NumPy) implementation
and a single `a2-highgpu-8g` (i.e., with 8 A100 cards) instance to test the GPU
(CuPy) implementation.
The training data was generated from the Lorenz96 model
[@lorenz_predictability_1996] with dimensions
$N_u=\{16,256\}$,
and we generated 80,000 total samples in the training dataset.

In the CPU tests, wall time scales quadratically with the reservoir size, while
it is mostly constant on a GPU.
For this problem, it becomes advantageous to use GPUs once the reservoir size is
approximately $N_r=8,000$ or greater.
In both the CPU and GPU tests, memory scales quadratically with reservoir size,
although the increasing memory usage with reservoir size is more dramatic on the
CPU than GPU.
This result serves as a motivation for our parallelized architecture.

## Parallel (Lazy) Architecture Strong Scaling Results

In order to evaluate the performance of the parallelized architecture, we take
the Lorenz96 system with dimension $N_u=256$ and subdivide the domain into
$N_g = \{2, 4, 8, 16, 32\}$ groups.
We then fix the problem size such that $N_r*N_g = 16,000$, so that
the timing results reflect strong scaling.
That is, the results show how the code performs with increasing resources on a fixed problem
size, which in theory correspond to Amdahl's Law [@amdahl_1967].
The training task and resources used are otherwise the same as for the standard
ESN results shown in \autoref{fig:eager}.
We then create 3 different `dask.distributed` Clusters, testing:

1. Purely threaded mode (CPU only).
2. The relevant default "LocalCluster" (i.e., single node) configuration for our resources.
   On the CPU resource, a GCP `c2-standard-60` instance,
   the default
   `dask.distributed.LocalCluster` has 6 workers, each with 5 threads.
   On the GPU resource, a GCP `a2-highgpu-8g` instance, the default
   `dask_cuda.LocalCUDACluster` has 8 workers, each
   with 1 thread.
3. A `LocalCluster` with 1 `dask` worker per group. On GPUs, this assumes 1 GPU per worker
   and we are able to use a maximum of 8 workers due to our available resources.

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
wall time with the standard (serial) architecture to the lazy (parallel)
architecture with $N_g$ groups.
On CPUs, using 1 `dask` worker process per ESN group generally scales well,
which makes sense because each group is trained entirely independently.

On GPUs, the timing is largely determined by how many workers (GPUs) there are
relative to the number of groups.
When the number of workers is less than the number of groups, performance is
detrimental.
However, when there is at least one worker per group, the timing is almost the
same as for the single worker case, only improving performance by 10-20%.
While the strong scaling is somewhat muted, the invariance of wall time to
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
The authors thank the editor Jonny Saunders for comments that
significantly improved the manuscript, and the reviewers Troy Arcomano and
William Nicholas.

# References
