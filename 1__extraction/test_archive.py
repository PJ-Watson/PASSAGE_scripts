"""An example workflow for reducing NIRISS/WFSS data from GLASS-JWST ERS."""

import argparse
import os
import pickle
import shutil
import tomllib
from pathlib import Path
from time import time

import numpy as np
from astropy.table import Table

# parser = argparse.ArgumentParser(description="Run the data extraction pipeline.")
# parser.add_argument(
#     "config_path",
#     type=str,
#     metavar="config_path",
#     help="The path of the configuration file used to setup the fits.",
# )
# args = parser.parse_args()

with open(
    "/media/sharedData/python/py3.13_PIE/code/PASSAGE_scripts/1__extraction/config_files/CINECA_config_par682.toml",
    "rb",
) as f:
    config = tomllib.load(f)


if __name__ == "__main__":

    from niriss_tools.pipeline.utils import gen_linefinding_outputs

    grizli_home_dir = Path(
        "/media/sharedData/data/2026_05_20_parallel_fields/2.0.2/passage-par682/grizli_home"
    )

    gen_linefinding_outputs(grizli_home_dir=grizli_home_dir)
