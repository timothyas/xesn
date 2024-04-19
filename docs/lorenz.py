try:
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()
    _use_cupy = True

except ImportError:
    _use_cupy = False

except xp.cuda.runtime.CUDARuntimeError:
    _use_cupy = False

import numpy as np
import xarray as xr
import cupy_xarray
from scipy.integrate import odeint

class Lorenz96():
    """A simple class for creating some sample data"""
    def __init__(
        self,
        N,
        forcing_term=8.,
        delta_t=0.01,
    ):
        self.N = N
        self.forcing_term = forcing_term
        self.delta_t = delta_t

    def rhs(self, x, t):
        """Lorenz 96 tendency equations

        Args:
            x (array_like): system state at one point in time
            t (float): time, not used here but passed as argument for odeint

        Returns:
            dx/dt (array_like): the temporal tendency
        """

        y = np.zeros(self.N)

        y[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        y[1] = (x[2] - x[self.N-1]) * x[0] - x[1]
        y[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]

        y[2:self.N-1] = (x[3:self.N] - x[:self.N-3]) * x[1:self.N-2] - x[2:self.N-1]

        y += self.forcing_term
        return y


    def generate(self, n_steps, x0=None):
        """Generate a time series of Lorenz 96 model output

        Args:
            n_steps (int): number of time steps to integrate forward
            x0 (array_like, optional): initial conditions for the system state

        Returns:
            xda (xarray.DataArray): with the full time series at increments of :attr:`delta_t`
        """

        x0 = np.zeros(self.N) if x0 is None else x0

        time = np.linspace(0., self.delta_t*n_steps, n_steps+1)
        values = odeint(self.rhs, x0, time)

        xda = xr.DataArray(
            data=values.T,
            coords={"x":np.arange(self.N),
                    "time":time},
            dims=("x","time"),
            attrs={
                "forcing_term": self.forcing_term,
                "delta_t": self.delta_t,
                "description": "Lorenz96 trajectory",
            }
        )
        xda["time"].attrs["delta_t"] = self.delta_t

        if _use_cupy:
            xda = xda.as_cupy()
        return xda
