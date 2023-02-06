try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    _use_cupy = True
except cp.cuda.runtime.CUDARuntimeError:
    cp = None
    _use_cupy = False

from test_gpu import ScalingTest

if __name__ == "__main__":


    n_system = 8

    pstr = "gpu" if _use_cupy else "cpu"
    for n_reservoir in [500, 1000, 2000, 4000, 8000, 16000]:
        test = ScalingTest(
                n_system=n_system,
                n_reservoir=n_reservoir,
                out_dir=f"gcp-{pstr}-test/{n_reservoir:06d}nr-{n_system:03d}ns")
        test()
