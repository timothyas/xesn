import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from xesn import Driver, _use_cupy

def sampler_to_xarray(ms, name):

    table = ms.to_pandas()
    time = [np.datetime64(x) for x in table.index]
    delta_t = time - time[0]
    vals = [x for x in table[name].values]
    return xr.DataArray(
        vals,
        coords={"delta_t": delta_t},
        dims=("delta_t",),
        name=name,
    )

def run_scaling_test(mode, n_input, n_reservoir, n_x):

    if mode == "lazy":
        client = Client()
        ms = MemorySampler()

    config_filename = f"config-{mode}.yaml"
    pstr = "az-gpu" if _use_cupy else "gcp-cpu"
    output_directory = f"{pstr}-{mode}/{n_reservoir:05d}nr-{n_input:03d}ni"

    driver = Driver(config=config_filename, output_directory=output_directory)
    driver.overwrite_config(
        {
            "xdata": {
                "zstore_path": f"lorenz96-{n_input:03d}d/trainer.zarr",
            },
            driver.esn_name: {
                "n_reservoir": n_reservoir,
                "n_input": n_input,
                "n_output": n_input,
            },
        }
    )

    if mode == "lazy":
        driver.overwrite_config({driver.esn_name: {"esn_chunks": {"x": n_x}}})
        with ms.sample("training"):
            driver.run_training()
    else:
        driver.run_training()

    if mode == "lazy":
        xmem = sampler_to_xarray(ms, name=f"training")
        xmem = xmem.expand_dims({
            "proc": ["gpu"] if _use_cupy else ["cpu"],
            "mode": [mode],
            "n_input": [n_input],
            "n_reservoir": [n_reservoir],
            "n_x": [n_x],
        })
        xmem = xmem.to_dataset()
        xmem.to_netcdf(f"{output_directory}/memory.nc")


if __name__ == "__main__":

    mode = "eager"
    n_x = np.nan
    for n_input in [16, 256]:
        for n_reservoir in [500]:#, 1_000, 2_000, 4_000, 8_000, 16_000]:
            run_scaling_test(
                mode=mode,
                n_input=n_input,
                n_reservoir=n_reservoir,
                n_x=n_x,
            )
