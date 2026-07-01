import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
from grizli.utils_numba import interp as interp_numba
from grizli.utils_numba import disperse as disperse_numba
from grizli import utils
import numpy as np
import matplotlib.pyplot as plt

# Used to test both utils_numba and utils_c
DISPERSE_MODULES = [disperse_numba]
INTERP_MODULES = [interp_numba]

if __name__=="__main__":

    # Default xarr precision
    rng = np.random.default_rng(1)

    # for orig_len, new_len in zip([20000, 100],[10,101]):
    #     # print (orig_len, new_len)
    #     # if orig_len>1e4:
    #     #     continue
    #     # continue
    #     # xarr = np.arange(1.0, 3.0, 0.0001)
    #     xarr = np.linspace(1.0, 3.0, orig_len)
    #     # print (xarr)
    #     # continue
    #     yarr = (np.abs(xarr - 2.0) <= 0.1) * 1.0
    #     # yarr = 0.5-2*(2.-xarr)**2
    #     # yarr[yarr<0.]=0.

    #     # np.random.seed(1)
    #     # xlr = rng.random(10) * 2 + 1
    #     xlr = rng.random(new_len) * 2 + 1
    #     xlr.sort()
    #     # print (xarr, xlr)

    #     for base_module in INTERP_MODULES:
    #         ylr = base_module.interp_conserve_c(xlr, xarr, yarr)

    #         # fig, axs = plt.subplots()
    #         # axs.plot(xarr, yarr)
    #         # axs.plot(xlr, ylr)

    #         # plt.show()

    #         # print (utils.trapz(ylr, xlr), utils.trapz(yarr, xarr))

    #     rng = np.random.default_rng(1)

    for orig_len, new_len in zip([20000, 100],[10,99]):
        xarr = np.linspace(1.0, 3.0, orig_len)
        yarr = (np.abs(xarr - 2.0) <= 0.1) * 1.0

        # np.random.seed(1)
        xlr = rng.random(new_len) * 2 + 1
        xlr.sort()

        for base_module in INTERP_MODULES:
            ylr = base_module.interp_conserve_c(xlr, xarr, yarr)

            print (f"{utils.trapz(ylr, xlr)=:.5f}", f"{utils.trapz(yarr, xarr)=:.5f}")
            # assert np.allclose(utils.trapz(ylr, xlr), utils.trapz(yarr, xarr))

    exit()


    xarr = np.array([0.0, 1.0, 2.0])
    yarr = np.array([0.0, 1.0, 0.0])
    for base_module in INTERP_MODULES:
        result = base_module.interp_c(np.array([0.5]), xarr, yarr)
        # assert np.allclose(result, 0.5)
        print (result)

    xarr = np.linspace(0.,2.0, 10000)
    yarr = (np.abs(xarr - 1.0) <= 0.1) * 1.0
    
    for base_module in INTERP_MODULES:
        result = base_module.interp_c(np.linspace(2.1,2.5,10001), xarr, yarr)


        print (result)

    sh = (128, 128)
    x0 = (32, 38)
    
    yp, xp = np.indices(sh)
    
    flam = np.sqrt((xp - x0[0]) ** 2 + (yp - x0[1]) ** 2).astype(np.float32)
    flam_neg = flam.copy()-4
    
    Rsize = 5
    segm = (flam < Rsize).astype(np.float32)

    total_flux = flam[segm > 0].sum()

    for disperse_module in DISPERSE_MODULES:

        # _ = disperse_module.compute_segmentation_limits(
        #     segm, 1.0, flam, np.array(sh, dtype=int)
        # )

        # imin, imax, ic, jmin, jmax, jc, area, tot_i = _

        # assert imin == (x0[1] - (Rsize - 1))
        # assert imax == (x0[1] + (Rsize - 1))
        # assert jmin == (x0[0] - (Rsize - 1))
        # assert jmax == (x0[0] + (Rsize - 1))
        # assert area == int(segm.sum())
        # assert np.allclose(tot_i, total_flux, rtol=1.0e-5)

        # _ = disperse_module.compute_segmentation_limits(
        #     segm, 2.0, flam, np.array(sh, dtype=int)
        # )
        # print (_)
        # imin, imax, ic, jmin, jmax, jc, area, tot_i = _
        # assert area == 0
        # assert np.allclose(tot_i, -99.0, rtol=1.0e-5)

        _ = disperse_module.compute_segmentation_limits(
            segm, 1.0, flam_neg, np.array(sh, dtype=int)
        )
        print (_)
        imin, imax, ic, jmin, jmax, jc, area, tot_i = _
        # assert area == int(segm.sum())
        # assert np.allclose(tot_i, -99.0, rtol=1.0e-5)

        # assert imin<=ic<=imax
        # assert jmin<=jc<=jmax

    # xarr = np.linspace(0.,2.0, 100)
    # yarr = (np.abs(xarr - 1.0) <= 0.1) * 1.0

    # for base_module in INTERP_MODULES:
    #     result = base_module.interp_c(np.linspace(-1.0,1.5,10), xarr, yarr)


    #     print (result)
    