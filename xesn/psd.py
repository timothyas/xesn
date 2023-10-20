import numpy as np
import xarray as xr
from scipy.stats import binned_statistic
from scipy.fft import fft, fft2, fftfreq

def psd_1d(xda):
    """Compute the 1D Power Spectral Density of a 1D array, varying in time.

    Note:
        This function is not lazy, and will call the entire array into memory before computing the PSD.

    Args:
        xda (xarray.DataArray): with "time" as final dimension

    Returns:
        xda_hat (xarray.DataArray): with 1D PSD along new, nondimensional 'k1d' dimension
    """

    assert xda.dims[-1] in ("time", "ftime"), "psd_1d requires 'time' to be on the last dimension"
    assert xda.ndim == 2

    n_x = xda.shape[0]
    n_time = xda.shape[-1]

    # transform and get amplitudes
    f_hat = fft(xda.values.T)
    psd = np.abs(f_hat)**2

    # get frequencies
    n_k = n_x//2 + 1
    k = n_x * fftfreq(n_x)

    # go 1D
    k_bins = np.arange(.5, n_k, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])

    psd_1d = np.zeros((n_x//2, n_time))
    for n in range(n_time):
        tmp1d, *_ = binned_statistic(
            k,
            psd[n],
            statistic="mean",
            bins=k_bins,
        )
        psd_1d[...,n] = tmp1d * np.pi * (k_bins[1:]**2 - k_bins[:-1]**2)
    xda_hat = _xpack(xda, k_vals, psd_1d)
    return xda_hat


def psd_2d(xda):
    """Compute the 1D Power Spectral Density of a 2D or 3D array, varying in time (so 3 or 4 axes total).
    If there are 3 non-time dimensions, then average over 3rd spatial dimension (e.g., depth).

    Note:
        This function is not lazy, and will call the entire array into memory before computing the PSD.

    Args:
        xda (xarray.DataArray): with "time" as final dimension

    Returns:
        xda_hat (xarray.DataArray): with 1D PSD along new, nondimensional 'k1d' dimension
    """

    assert xda.dims[-1] in ("time", "ftime"), "psd_2d requires 'time' to be on the last dimension"
    try:
        assert xda.ndim < 5
    except AssertionError:
        raise NotImplementedError("psd_2d will only work for up to 3D time varying arrays (i.e., 4 dims total)")


    n_x = xda.shape[0]
    n_time = xda.shape[-1]

    # transform and get amplitudes
    f_hat = fft2(xda.values.T)
    psd = np.abs(f_hat)**2

    # get frequencies
    n_k = n_x//2 + 1
    k = n_x * fftfreq(n_x)
    k2d = np.meshgrid(k, k)
    k_tot = np.sqrt(k2d[0]**2 + k2d[1]**2)

    # go 1D
    k_bins = np.arange(.5, n_k, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])

    psd_1d = np.zeros((n_x//2, n_time))
    for n in range(n_time):
        arr2d = psd[n,...].mean(axis=0) if psd.ndim>3 else psd[n]
        tmp1d, *_ = binned_statistic(
            k_tot.flatten(),
            arr2d.flatten(),
            statistic="mean",
            bins=k_bins,
        )
        psd_1d[...,n] = tmp1d * np.pi * (k_bins[1:]**2 - k_bins[:-1]**2)


    xda_hat = _xpack(xda, k_vals, psd_1d)
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
