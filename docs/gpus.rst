Notes on GPU Implementation
###########################

The eager and lazy ESN frameworks are currently implemented using
`cupy <https://cupy.dev/>`_, with the
`cupy-xarray <https://cupy-xarray.readthedocs.io/>`_
API.
See `these installation instructions <installation.rst#Using-GPUs>`_
for setting up to use GPUs.
There are only small changes that need to be made to the
CPU-based examples in order to run them on a GPU.

Required Changes and Tips Using Standard and Parallel ESNs
----------------------------------------------------------

1. Put the training and testing data on the GPU by running ``dataset.as_cupy()``
   after they have been created. Following the example notebooks, this would
   look like

   .. code-block:: python

      trainer = trainer.as_cupy()
      tester = tester.as_cupy()

   after these datasets have been created.
   Note that this is taken care of automatically by using the
   :class:`xesn.Driver` class.

2. The eigenvalue-based ESN adjacency matrix normalization method (i.e.,
   spectral radius) can no longer be used, so this option must be set to
   ``adjacency_kwargs={"normalization": "svd"}`` or more simply
   ``adjacency_kwargs={"normalization": "multiply"}``.

3. Performance is typically much worse with the default ``"coo"`` sparse matrix
   format, and so it is recommended to use
   ``adjacency_kwargs={"format": "csr"}`` when creating the ESN.

4. Before writing the weights out, the data has to be pulled to the CPU as
   follows with an xarray dataset ``xds``:

   .. code-block:: python

      xds.as_numpy().to_zarr("esn-weights.zarr")



Not Implemented: Macro Optimization
-----------------------------------

Unfortunately, GPU/cupy integration is not currently implemented in the
`Surrogate Modeling Toolbox <https://smt.readthedocs.io/en/latest>`_
and so the Macro Optimization cannot be performed on GPUs.
