try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    _use_cupy = True
except cp.cuda.runtime.CUDARuntimeError:
    cp = None
    _use_cupy = False

from test_gpu import ScalingTest

if __name__ == "__main__":


    pstr = "gpu" if _use_cupy else "cpu"
    n_system = 128

    batches = {
            500     : 100_000,
            1_000   : 100_000,
            2_000   :  50_000,
            4_000   :  25_000,
            8_000   :  12_500,
            16_000  :   6_250}

    for n_reservoir in [500,   1_000,  2_000,  4_000,  8_000, 16_000]:
        test = ScalingTest(
                n_system=n_system,
                n_reservoir=n_reservoir,
                batch_size=batches[n_reservoir],
                out_dir=f"gcp-{pstr}-test/{n_reservoir:06d}nr-{n_system:03d}ns")
        test()
