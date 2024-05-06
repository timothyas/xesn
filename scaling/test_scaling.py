import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from shutil import rmtree
from distributed import Client
from distributed.diagnostics import MemorySampler

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

@profile
def run_scaling_test(mode, n_input, n_reservoir, n_x):


    if "lazy" in mode:
        ms = MemorySampler()
        if "threaded" in mode:
            client = Client(processes=False)
        elif "worker-default" in mode:
            client = Client()
        else:
            n_workers = n_input // n_x
            client = Client(n_workers=n_workers)


    config_filename = f"config-{mode.replace('-threaded','').replace('-worker-default','')}.yaml"
    pstr = "az-gpu" if _use_cupy else "gcp-cpu"
    output_directory = f"{pstr}-{mode}/{n_reservoir:05d}nr-{n_input:03d}ni"
    if n_x is not None:
        output_directory += f"-{n_x:02d}nx"

    # if lazy, rechunk the data
    if "lazy" in mode:
        xds = xr.open_zarr(f"lorenz96-{n_input:03d}d/trainer.zarr")
        xds["trajectory"].encoding={}
        xds = xds.chunk({"x": 2, "time": -1})
        xds.to_zarr("scaling-dataset.zarr", mode="w")

    driver = Driver(config=config_filename, output_directory=output_directory)
    driver.overwrite_config(
        {
            "xdata": {
                "zstore_path": f"scaling-dataset.zarr",
            },
            driver.esn_name: {
                "n_reservoir": n_reservoir,
            },
        }
    )

    if "lazy" in mode:
        driver.overwrite_config({driver.esn_name: {"esn_chunks": {"x": n_x}}})
        with ms.sample("training"):
            driver.run_training()
    else:
        driver.overwrite_config({driver.esn_name: {"n_input": n_input, "n_output": n_output}})
        driver.run_training()

    if "lazy" in mode:
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
        rmtree("scaling-dataset.zarr")


if __name__ == "__main__":

    #mode = "eager"
    #n_x = np.nan
    #for n_input in [16, 256]:
    #    for n_reservoir in [500, 1_000, 2_000, 4_000, 8_000, 16_000]:
    #        run_scaling_test(
    #            mode=mode,
    #            n_input=n_input,
    #            n_reservoir=n_reservoir,
    #            n_x=n_x,
    #        )

    mode = "lazy-threaded"
    n_input = 256
    for n_reservoir, n_x in zip(
        [8_000, 4_000, 2_000, 1_000, 500, 250],
        [  128,    64,    32,    16,   8,   4],
        ):
        run_scaling_test(
            mode=mode,
            n_input=n_input,
            n_reservoir=n_reservoir,
            n_x=n_x,
        )
