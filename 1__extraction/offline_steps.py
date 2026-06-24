"""Run all steps requiring a network connection before processing."""

import argparse
import os
import tomllib
from pathlib import Path

import numpy as np
from astropy.table import Table

parser = argparse.ArgumentParser(
    description="Run all steps requiring a network connection before processing."
)
parser.add_argument(
    "config_path",
    type=str,
    metavar="config_path",
    help="The path of the configuration file used to setup the fits.",
)

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.config_path, "rb") as f:
        config = tomllib.load(f)

    # Latest context
    os.environ["CRDS_CONTEXT"] = (
        f"jwst_{config["calibrations"].get("crds_ver", 1535)}.pmap"
    )
    # Set to "NGDEEP" to use those calibrations
    os.environ["NIRISS_CALIB"] = config["calibrations"].get(
        "niriss_calib", "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"
    )

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
    from niriss_tools.pipeline import (
        construct_exposure_table,
        gaia_catalogue_from_obs_table,
        queryMAST,
    )

    root_dir = Path(os.path.expandvars(config["general"].get("root_dir", Path.cwd())))
    field = config["general"].get("field")

    field_name = f"{config["general"].get("field_prefix")}-{field}".lower()

    field_obs_IDs = config["general"].get("field_obs_ids")
    proposal_IDs = config["general"].get("proposal_ids")

    reduction_ver = config["general"].get("reduction_ver", "1.0.0")

    reduction_dir = root_dir / reduction_ver / field_name
    reduction_dir.mkdir(exist_ok=True, parents=True)

    level_1_dir = reduction_dir / "Level1"

    if not config["general"].get("skip_stage_1", True):

        # Find the correct observations
        if not (root_dir / f"MAST_summary_{proposal_IDs}.csv").is_file():
            all_obs_tab = queryMAST(proposal_IDs)
            all_obs_tab.write(
                root_dir / f"MAST_summary_{proposal_IDs}.csv", overwrite=True
            )
        else:
            all_obs_tab = Table.read(root_dir / f"MAST_summary_{proposal_IDs}.csv")

        field_obs_tab = all_obs_tab[np.isin(all_obs_tab["obs_id_num"], field_obs_IDs)]

        from mastquery import utils as mastutils

        MAST_dir = reduction_dir / "MAST_downloads"
        MAST_dir.mkdir(exist_ok=True, parents=True)

        level_1_dir.mkdir(exist_ok=True, parents=True)

        field_obs_download = field_obs_tab[
            ~np.asarray(
                [
                    (level_1_dir / f"{s}_rate.fits").is_file()
                    for s in field_obs_tab["obs_id"]
                ],
                dtype=bool,
            )
        ]

        if len(field_obs_download) > 0:
            mastutils.download_from_mast(field_obs_download, path=MAST_dir)

        all_exp_tab = construct_exposure_table(MAST_dir, ext_pattern="*uncal.fits")

    else:
        all_exp_tab = construct_exposure_table(level_1_dir, ext_pattern="*rate.fits")

    direct_tab = all_exp_tab[["CLEAR" in c for c in all_exp_tab["filter"]]]

    gaia = gaia_catalogue_from_obs_table(direct_tab)
    gaia.write(reduction_dir / f"{field_name}.gaia.fits")

    from grizli.prep import table_to_radec, table_to_regions

    table_to_radec(gaia[gaia["valid"]], reduction_dir / f"{field_name}.gaia.radec")
    table_to_regions(gaia[gaia["valid"]], reduction_dir / f"{field_name}.gaia.reg")
