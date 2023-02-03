import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from ddc import DataLorenz96

import sys
sys.path.append("..")
from esnpy import ESN, from_zarr


def generate_data(
        system_dimension=6,
        n_train=42_000,
        n_test=1_001,
        n_spinup=500):


    data = DataLorenz96(system_dimension=system_dimension)
    data.generate(n_steps=n_train)
    trainer = data.values.copy()

    data.generate(n_steps=1_000)
    data.generate(n_steps=n_test+n_spinup)
    tester = data.values.copy()
    return cp.asarray(trainer), cp.asarray(tester)


if __name__ == "__main__":

    system_dimension = 6

    esn = ESN(
        n_input=system_dimension,
        n_output=system_dimension,
        n_reservoir=500,
        input_factor=0.863,
        adjacency_factor=0.713,
        connectedness=5,
        bias=1.76,
        leak_rate=0.874,
        tikhonov_parameter=6.9e-7,
        input_kwargs={
            "normalization": "svd",
        },
        adjacency_kwargs={
            "normalization": "svd",
        },
        random_seed=0,
    )

    trainer, tester = generate_data(system_dimension=system_dimension)

    esn.build()
    esn.train(trainer)
    y = esn.predict(tester, n_steps=1_000, n_spinup=500)
    u = tester[:, 500:]

    time = 0.01 * np.arange(y.shape[-1])


    nrows = system_dimension
    fig, axs = plt.subplots(nrows, 1, figsize=(8, nrows*1), constrained_layout=True, sharex=True, sharey=True)

    for ui, yi, ax in zip(u.get(), y.get(), axs):
        ax.plot(time, ui, color='k')
        ax.plot(time, yi, color="C2")
        for key in ["top", "right"]:
            ax.spines[key].set_visible(False)

    fig.savefig("esn-prediction.pdf", bbox_inches="tight")
