import numpy as np
from lorenz import Lorenz96

def create_dataset(n_input, n_spinup, n_train, n_transient, n_test, random_seed=None):

    model = Lorenz96(N=n_input)

    n_total = n_spinup + n_train + n_transient + n_test

    rs = np.random.RandomState(random_seed)
    x0 = np.zeros(n_input)
    x0[0] = 0.01
    trajectory = model.generate(n_steps=n_total, x0=x0)

    trajectory = trajectory.to_dataset(name="trajectory")
    trainer = trajectory.isel(time=slice(n_spinup, n_spinup+n_train+1))
    tester = trajectory.isel(time=slice(-n_test-1, None))
    bias = trainer.mean()
    scale = trainer.std()
    trainer = (trainer - bias) / scale
    tester = (tester - bias) / scale
    return trainer, tester

if __name__ == "__main__":

    for n_input in [16, 256]:
        trainer, tester = create_dataset(
            n_input=n_input,
            n_spinup=20_000,
            n_train=80_000,
            n_transient=10_000,
            n_test=10_000,
        )
        trainer.to_zarr(f"lorenz96-{n_input:03d}d/trainer.zarr")
        trainer.to_zarr(f"lorenz96-{n_input:03d}d/tester.zarr")
