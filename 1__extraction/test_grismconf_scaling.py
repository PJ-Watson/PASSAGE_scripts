"""An example workflow for reducing NIRISS/WFSS data from GLASS-JWST ERS."""

import os
from pathlib import Path

import numpy as np

try:
    from passagepipe import utils
except:
    import utils

from astropy.table import Table

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1535.pmap"
# Set to "NGDEEP" to use those calibrations
# os.environ["NIRISS_CALIB"] = "CONF/CUSTOM/COMBINE_STSCI_GRIZLI_{1}_{0}_V1.conf"

# Symlink the custom configuration files.
# The format for these is pretty self explanatory, and the current set
# combines the NGDEEP configuration for the first order, with the grizli defaults
# for all other orders.
for orig in (Path(__file__).parent / "conf_data").glob("*"):
    if not (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).exists():
        (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).symlink_to(
            orig, target_is_directory=orig.is_dir()
        )

# https://github.com/PJ-Watson/niriss-tools
from niriss_tools import pipeline

# Change this to reduce a different field
root_dir = Path(os.getenv("ROOT_DIR"))
# root_dir = Path("/media/watsonp/ArchivePJW/backup/data")
field = "Par08"
date = "2026_05_08"

field_name = f"lcs-{field.lower()}"

# try:
#     passage_tab = Table.read(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")
# except:
#     import pandas as pd

#     df = pd.read_csv(root_dir / "JWST PASSAGE Cycle 1 - Cy1 Executed.csv")
#     df = df[["Par#", "Obs Date", "PID", "Obs ID", "Filter", "Mode"]]
#     df = df.ffill()
#     df = df[~df["Obs Date"].str.contains("SKIPPED")]
#     # print (df)
#     passage_tab = Table.from_pandas(df)
#     passage_tab.write(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")

# # exit()

# field_obs_IDs = list(passage_tab[passage_tab["Par#"] == field]["Obs ID"])
# proposal_ID = list(passage_tab[passage_tab["Par#"] == field]["PID"])[0]

field_obs_IDs = [2]
proposal_ID = 9365

# reduction_dir = Path(root_dir) / f"{date}_{field_name}"
reduction_dir = Path(root_dir) / f"{date}_PIE"
reduction_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":

    from grizli.grismconf import load_grism_config
    import matplotlib.pyplot as plt

    conf = load_grism_config(
        str(
            root_dir
            / "/misc_data_conf/crds_cache/references/jwst/niriss/jwst_niriss_specwcs_0083.asdf"
        )
    )

    print (dir(conf))
    print (conf.sens["B"])

    print (conf.dxlam)
    print ([f"{k}: {len(v)}" for k, v in conf.dxlam.items()])
    print (conf.nx)

    # plt.plot(conf.sens["B"]["WAVELENGTH"], conf.sens["B"]["SENSITIVITY"])
    # plt.show()

    beam = "B"

    xc = 1024
    yc = 1024
    xcenter = 0.
    ycenter = 0.
    pad = 0.
    pad = [0.,0.]
    grow = 1
    fwcpos = None
    MW_F99 = None

    from grizli.utils_numba import interp

    dx = conf.dxlam[beam]  # + xcenter #-xoffset

    dx = conf.eval_dxlam(
        x=(xc + xcenter - pad[1]) / grow,
        y=(yc + ycenter - pad[0]) / grow,
        beam=beam,
    )

    # print (dx, dx_eval)
    # exit()

    if grow > 1:
        dx = np.arange(dx[0] * grow, dx[-1] * grow)

    xoffset = 0.0

    if ("G14" in conf.conf_file) & (beam == "A"):
        xoffset = -0.5  # necessary for WFC3/IR G141, v4.32

    # xoffset = 0. # suggested by ACS
    # xoffset = -2.5 # test

    xoffset = xoffset
    ytrace_beam, lam_beam = conf.get_beam_trace(
        x=(xc + xcenter - pad[1]) / grow,
        y=(yc + ycenter - pad[0]) / grow,
        dx=(dx + xcenter * 0 + xoffset) / grow,
        beam=beam,
        fwcpos=fwcpos,
    )

    ytrace_beam *= grow

    # Integer trace
    # Add/subtract 20 for handling int of small negative numbers
    dyc = np.asarray(ytrace_beam + 20, dtype=int) - 20 + 1

    # Account for pixel centering of the trace
    yfrac_beam = ytrace_beam - np.floor(ytrace_beam)

    # Interpolate the sensitivity curve on the wavelength grid.
    ysens = lam_beam * 0
    so = np.argsort(lam_beam)

    conf_sens = conf.sens[beam]
    if MW_F99 is not None:
        MWext = 10 ** (-0.4 * (MW_F99(conf_sens["WAVELENGTH"] * u.AA)))
    else:
        MWext = 1.0

    ysens[so] = interp.interp_conserve_c(
        lam_beam[so],
        conf_sens["WAVELENGTH"],
        conf_sens["SENSITIVITY"] * MWext,
        integrate=1,
        left=0,
        right=0,
    )
    lam_sort = so

    # Needs term of delta wavelength per pixel for flux densities
    # dl = np.abs(np.append(lam_beam[1] - lam_beam[0],
    #                     np.diff(lam_beam)))
    # ysens *= dl#*1.e-17
    sensitivity_beam = ysens

    plt.plot(dx, ysens)
    plt.show()