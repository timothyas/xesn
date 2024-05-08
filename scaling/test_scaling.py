import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from shutil import rmtree
from distributed import Client
from distributed.diagnostics import MemorySampler

from xesn import Driver, _use_cupy

if _use_cupy:
    from dask_cuda import LocalCUDACluster


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
        if "threaded" in mode:
            client = Client(processes=False)
        elif "worker-default" in mode:
            if _use_cupy:
                cluster = LocalCUDACluster()
                client = Client(cluster)
            else:
                client = Client()

        else:
            n_workers = n_input // n_x

            if _use_cupy:
                cvd = str([x for x in range(min(n_workers, 8))])[1:-1].replace(" ", "")
                cluster = LocalCUDACluster(
                    CUDA_VISIBLE_DEVICES=cvd,
                )
                client = Client(cluster)

            else:
                client = Client(n_workers=n_workers)
        ms = MemorySampler()

    config_filename = f"config-{mode.replace('-threaded','').replace('-worker-default','')}.yaml"
    pstr = "gcp-gpu" if _use_cupy else "gcp-cpu"
    output_directory = f"{pstr}-{mode}/{n_reservoir:05d}nr-{n_input:03d}ni"
    if n_x is not None:
        output_directory += f"-{n_x:02d}nx"

    # if lazy, rechunk the data
    zstore_path = f"lorenz96-{n_input:03d}d/trainer.zarr"
    if "lazy" in mode:
        xds = xr.open_zarr(zstore_path)
        xds["trajectory"].encoding={}
        xds = xds.chunk({"x": 2, "time": -1})
        xds.to_zarr("scaling-dataset.zarr", mode="w")
        zstore_path = "scaling-dataset.zarr"

    driver = Driver(config=config_filename, output_directory=output_directory)
    driver.overwrite_config(
        {
            "xdata": {
                "zstore_path": zstore_path,
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
        driver.overwrite_config({driver.esn_name: {"n_input": n_input, "n_output": n_input}})
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

    if "lazy" in mode:
        client.shutdown()

if __name__ == "__main__":

    #mode = "eager"
    #n_x = None
    #for n_input in [16, 256]:
    #    for n_reservoir in [500, 1_000, 2_000, 4_000, 8_000, 16_000]:
    #       run_scaling_test(
    #           mode=mode,
    #           n_input=n_input,
    #           n_reservoir=n_reservoir,
    #           n_x=n_x,
    #       )

    mode = "lazy"
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
