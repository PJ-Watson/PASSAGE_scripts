import matplotlib.pyplot as plt

import numpy as np
from astropy.table import Table, join
from pathlib import Path

import plotutils
import sed_utils

import tomllib

plotutils.formatting.setup_aanda_style()

passage_dir = Path("/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data")

config_path = "/media/sharedData/python/py3.13_PIE/code/PASSAGE_scripts/3__SED_fitting/config_v0.2.1.toml"

field = "Par2744"
fit_ver = "v0.2.1"

if __name__ == "__main__":

    with open(config_path, "rb") as f:
        config = tomllib.load(f)


    pipes_params = sed_utils.correct_pipes_params(config["fit_instructions"])

    filter_list = np.loadtxt(
        passage_dir / field / f"{field}_filter_list_{fit_ver}.txt",
        dtype=str,
    )

    import pprint

    pprint.pprint(pipes_params)

    from bagpipes.fitting.check_priors import check_priors

    Check = check_priors(pipes_params, filt_list=filter_list, n_draws=10, phot_units="mujy")

    # print (Check.params, check.pdfs)
    for n, p in zip(Check.params, Check.pdfs):
        print (n, p)