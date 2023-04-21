# esnpy
A clean echo state network implementation

## Feature list

### ESN Architecture related

- [ ] Ensemble from initialized matrices
- [ ] Hessian ensemble
- [ ] Ability to overlap in more than 2 dimensions

### Drivers, and support

- [ ] For lazy data, is the time slicer necessary?
- [ ] YAMLParser, if necessary
- [ ] Timer
- [ ] Data generation vs Data accessing:
    - Here, assume that we're always accessing the data, rely on other repos to
      do the generation
- [ ] Can we do the data splitting with sci kit learn rather than my cooked up
  stuff?
    - Get indices
    - Slice the data
    - Preprocess
    - Train: Inner, predict, etc
    - Postprocess: Keep VPT calc, always store trajectories?
- [ ] Preprocessing:
    - Subset in space
    - Subsample in time
    - Rechunk
    - Add noise to the data
    - Normalization
    - Persist
- [ ] Dask handling:
    - Move some of the repeated/copy pasta'd code from my run scripts to the
      drivers... should be as abstract as possible to create client/cluster
- [ ] Cost function:
    - ... want to use surrogate modeling toolbox?
    - Spectral cost... should probably just let the user define this, and keep
      it in my runtime repo

### Testing, docs, etc

- [ ] Make sure environment yaml is good to go
- [ ] Unit tests
- [ ] Integration tests
- [ ] Regression tests
- [ ] Docstrings, demo notebooks

## Performance notes

- What is the data layout after converting to dask array? Still F in each chunk?
    * No, always converted to C format

Testing with 60 dim L96, 100,000 time steps, batch size 10,000

###  Speed with Eager

The input data layout matters very little... what matters more is the
reservoir layout.
  - With reservoir "space" in memory (so create rT): 10 sec
  - With reservoir time in memory (so create r, take transpose): 15 sec

With input data as either C or F contiguous, would mean 10.4 vs 9.6 seconds...
so slightly faster to have input data with spatial dimension in memory

### Speed with lazy

Creating the same as the eager data, stored to Zarr with either time first or
last, taking transpose with the former before feeding in.
With 2 reservoirs, took:
    - time stored last: 19.5 seconds
    - time stored first: 20.6 seconds

Not huge, but slightly faster to store as it is meant to be accessed.
