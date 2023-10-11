# Performance scratch notes

- What is the data layout after converting to dask array? Still F in each chunk?
    * No, always converted to C format

Testing with 60 dim L96, 100,000 time steps, batch size 10,000

##  Speed with Eager

The input data layout matters very little... what matters more is the
reservoir layout.
  - With reservoir "space" in memory (so create rT): 10 sec
  - With reservoir time in memory (so create r, take transpose): 15 sec

With input data as either C or F contiguous, would mean 10.4 vs 9.6 seconds...
so slightly faster to have input data with spatial dimension in memory

## Speed with lazy

Creating the same as the eager data, stored to Zarr with either time first or
last, taking transpose with the former before feeding in.
With 2 reservoirs, took:
    - time stored last: 19.5 seconds
    - time stored first: 20.6 seconds

Not huge, but slightly faster to store as it is meant to be accessed.
