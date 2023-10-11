import os
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    _use_cupy = True
except cp.cuda.runtime.CUDARuntimeError:
    cp = None
    _use_cupy = False

import numpy as np
import matplotlib.pyplot as plt
import logging
from contextlib import redirect_stdout

from ddc import DataLorenz96

from timer import Timer

import sys
sys.path.append("..")
from xesn import ESN, from_zarr

class ScalingTest():

    n_system    = None
    n_reservoir = None
    n_train     = 100_000
    n_test      = 2_000
    n_spinup    = 500
    batch_size  = None
    out_dir     = "scaling-results"

    def __init__(self, **kw):
        for key, val in kw.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)


        # setup logging
        os.makedirs(self.out_dir)
        self.logfile = os.path.join(self.out_dir, "stdout.log")

        # timers
        self.walltime = Timer(filename=self.logfile)
        self.localtime = Timer(filename=self.logfile)


    def __call__(self):

        self.walltime.start()
        self.localtime.start("Generate Data")
        trainer, tester = self.generate_data()
        self.localtime.stop()

        self.localtime.start("Create ESN")
        esn = self.get_esn()
        self.localtime.stop()

        self.localtime.start("Training")
        esn.train(trainer, batch_size=self.batch_size)
        self.localtime.stop("Training time")

        self.localtime.start("Prediction")
        y = esn.predict(tester, n_steps=self.n_test, n_spinup=self.n_spinup)
        self.localtime.stop("Prediction time")

        self.localtime.start("Store model")
        ds = esn.to_xds()
        ds.to_zarr(os.path.join(self.out_dir, "esn.zarr"))
        self.localtime.stop("IO time")

        self.walltime.stop("Total Walltime")


    def print_log(self, *args):
        with open(self.logfile, "a") as file:
            with redirect_stdout(file):
                print(*args)


    def generate_data(self):

        data = DataLorenz96(system_dimension=self.n_system)
        data.generate(n_steps=self.n_train)
        trainer = data.values.copy()

        data.generate(n_steps=1_000)
        data.generate(n_steps=self.n_test+self.n_spinup+1)
        tester = data.values.copy()
        if _use_cupy:
            returns = cp.asarray(trainer), cp.asarray(tester)
        else:
            returns = trainer, tester
        return returns


    def get_esn(self):

        esn = ESN(
            n_input=self.n_system,
            n_output=self.n_system,
            n_reservoir=self.n_reservoir,
            input_factor=0.86,
            adjacency_factor=0.71,
            connectedness=5,
            bias=1.76,
            leak_rate=0.87,
            tikhonov_parameter=6.9e-7,
            input_kwargs={
                "normalization": "svd",
            },
            adjacency_kwargs={
                "normalization": "svd",
            },
            random_seed=0,
        )
        esn.build()
        return esn


if __name__ == "__main__":


    n_system = 128

    batches = {
            500     : 100_000,
            1_000   : 100_000,
            2_000   :  50_000,
            4_000   :  25_000,
            8_000   :  12_500,
            16_000  :   6_250}

    pstr = "gpu" if _use_cupy else "cpu"

    for n_reservoir in [    500,   1_000,  2_000,  4_000,  8_000, 16_000]:
        batch_size = batches[n_reservoir] if n_system==128 else None
        test = ScalingTest(
                n_system=n_system,
                n_reservoir=n_reservoir,
                batch_size=batch_size,
                out_dir=f"psl-{pstr}-test/{n_reservoir:06d}nr-{n_system:03d}ns")
        test()
