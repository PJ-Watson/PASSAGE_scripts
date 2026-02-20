"""An example workflow for performing a multi-region fit to NIRISS data."""

import os
import tomllib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from niriss_tools.grism import MultiRegionFit

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1467.pmap"
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"

# https://github.com/PJ-Watson/niriss-tools
from niriss_tools import pipeline

# Change this to reduce a different field
root_dir = Path(os.getenv("ROOT_DIR"))
# root_dir = Path("/media/watsonp/ArchivePJW/backup/data")
field = "Par028"
date = "2026_02_14"

field_name = f"passage-{field.lower()}"

try:
    passage_tab = Table.read(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")
except:
    import pandas as pd

    df = pd.read_csv(root_dir / "JWST PASSAGE Cycle 1 - Cy1 Executed.csv")
    df = df[["Par#", "Obs Date", "PID", "Obs ID", "Filter", "Mode"]]
    df = df.ffill()
    df = df[~df["Obs Date"].str.contains("SKIPPED")]
    # print (df)
    passage_tab = Table.from_pandas(df)
    passage_tab.write(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")

# exit()

field_obs_IDs = list(passage_tab[passage_tab["Par#"] == field]["Obs ID"])
proposal_ID = list(passage_tab[passage_tab["Par#"] == field]["PID"])[0]

reduction_dir = Path(root_dir) / f"{date}_{field_name}"
reduction_dir.mkdir(exist_ok=True, parents=True)

config_path = Path(__file__).parent / "config_passage.toml"

if __name__ == "__main__":

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    galaxies = {
        578: 1.438,
        1779: 1.372,
        1997: 1.406,
        2056: 1.865,
        2275: 3.079,
        2993: 1.248,
        11118: 2.275,
        13154: 1.70,
        13356: 3.08,
        243: 1.247,
        10713: 0.891,
        721: 1.265,
        762: 2.277,
    }
    # flagged_cat.pprint()
    for obj_id, obj_z in galaxies.items():

        obj_z = round(
            fits.getheader(
                reduction_dir
                / "grizli_home"
                / "Extractions"
                / "full"
                / f"{field_name}_{obj_id:0>5}.full.fits"
            )["REDSHIFT"],
            5,
        )

        print(obj_id, obj_z)

        multiregion = MultiRegionFit(config_path, obj_id, obj_z, run_all=False)
        out_path = (
            multiregion.out_dir
            / "multiregion"
            / f"regions_{obj_id:0>5}_z_{obj_z}_{multiregion.grism_fit_kwargs["pline"].get("pixscale", 0.06)}arcsec.line.fits"
        )

        if not out_path.is_file():
            print(f"{out_path} not found, running fit for object {obj_id}.")
            multiregion.run_all()
