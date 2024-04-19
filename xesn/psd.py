import numpy as np
import xarray as xr

from . import _use_cupy
from scipy.stats import binned_statistic
from scipy.fft import fftfreq
if _use_cupy:
    from cupyx.scipy.fft import fft, fft2
else:
    from scipy.fft import fft, fft2

def psd(xda):
    """Compute the 1D Power Spectral Density of a 1D, 2D, or 3D array, varying in time (so 2-4 axes total).
    If there are 3 non-time dimensions, then average amplitudes over 3rd spatial dimension (e.g., depth) before binning.

    Note:
        This function is not lazy, and will call the entire array into memory before computing the PSD.
        Additionally, the final binning does not run on the GPU, so it will pull FFT data to the CPU, and put it back on the device.

    Args:
        xda (xarray.DataArray): with "time" as final dimension

    Returns:
        xda_hat (xarray.DataArray): with 1D PSD along new, nondimensional 'k1d' dimension
    """

    assert xda.dims[-1] in ("time", "ftime"), "psd_1d requires 'time' to be on the last dimension"
    try:
        assert xda.ndim < 5
    except AssertionError:
        raise NotImplementedError("psd will only work for up to 3D time varying arrays (i.e., 4 dims total)")

    n_x = xda.shape[0]
    n_time = xda.shape[-1]

    # transform and get amplitudes
    xda = xda.load()
    f_hat = fft(xda.data.T) if xda.ndim == 2 else fft2(xda.data.T)
    if _use_cupy:
        f_hat = f_hat.get()
    psi = np.abs(f_hat)**2

    # get frequencies
    n_k = n_x//2 + 1
    k = n_x * fftfreq(n_x)
    if xda.ndim == 2:
        k_tot = k
    else:
        k2d = np.meshgrid(k, k)
        k_tot = np.sqrt(k2d[0]**2 + k2d[1]**2)

    # go 1D
    k_bins = np.arange(.5, n_k, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])

    psi_1d = np.zeros((n_x//2, n_time))
    for n in range(n_time):
        arr = psi[n].mean(axis=0) if psi.ndim > 3 else psi[n]
        tmp1d, *_ = binned_statistic(
            k_tot.flatten(),
            arr.flatten(),
            statistic="mean",
            bins=k_bins,
        )
        psi_1d[...,n] = tmp1d * np.pi * (k_bins[1:]**2 - k_bins[:-1]**2)

    xda_hat = _xpack(xda, k_vals, psi_1d)
    if _use_cupy:
        xda_hat = xda_hat.as_cupy()
    return xda_hat


def _xpack(xda, k_vals, psd_1d):


    attrs = {}
    if "long_name" in xda.attrs:
        attrs["long_name"] = f"psd_of_"+xda.attrs["long_name"]

    tdim = "time" if "time" in xda.dims else "ftime"
    xda_hat = xr.DataArray(
            psd_1d,
            coords={
                "k1d": k_vals,
                "time": xda[tdim],
                },
            dims=("k1d", tdim),
            attrs=attrs,
        )
    xda_hat["k1d"].attrs={
            "units": "",
            "description": "nondimensional 1D wavenumber",
        }
    return xda_hat
