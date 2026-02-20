"""An example workflow for reducing NIRISS/WFSS data from GLASS-JWST ERS."""

import os
import shutil
from pathlib import Path

import numpy as np
from astropy.table import Table
from niriss_tools import pipeline

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1467.pmap"
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"

# Symlink the custom configuration files.
# The format for these is pretty self explanatory, and the current set
# combines the NGDEEP configuration for the first order, with the grizli defaults
# for all other orders.
for orig in (Path(__file__).parent / "conf_data").glob("*"):
    if not (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).exists():
        (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).symlink_to(
            orig, target_is_directory=orig.is_dir()
        )

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

z_cat = Table.read(
    root_dir
    / "2026_01_08__PASSAGE"
    / "PASSAGE_data"
    / "Par028"
    / "Par028_matched_cosmosweb.fits"
)

z_cat.pprint()

import warnings

# Quiet some of the common the grizli-induced warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astropy.wcs import FITSFixedWarning

for w in [VerifyWarning, FITSFixedWarning, UnitsWarning]:
    warnings.simplefilter("ignore", category=w)


if __name__ == "__main__":

    import logging

    import grizli
    from astropy.io import fits
    from grizli import fitting, jwst_utils, multifit, prep, utils
    from grizli.pipeline import auto_script

    print("Grizli version: ", grizli.__version__)

    # Quiet JWST log warnings
    jwst_utils.QUIET_LEVEL = logging.INFO
    jwst_utils.set_quiet_logging(jwst_utils.QUIET_LEVEL)

    # Setup the grizli directory structure
    grizli_home_dir = reduction_dir / "grizli_home"

    kwargs = auto_script.get_yml_parameters()

    # The number of processes to use
    cpu_count = 4

    prep_dir = grizli_home_dir / f"Prep"
    prep_dir.mkdir(exist_ok=True, parents=True)

    extractions_dir = grizli_home_dir / f"Extractions"
    extractions_dir.mkdir(exist_ok=True, parents=True)

    files_to_link = []
    patterns = [
        "*drc*.fits",
        "*_seg.fits",
        "*GrismFLT.fits",
        "*GrismFLT.pkl",
        "*wcs.fits",
        "*cat.fits",
        "*phot.fits",
    ]
    for p in patterns:
        files_to_link.extend(prep_dir.glob(p))
    for file in files_to_link:
        if not (extractions_dir / file.name).is_file():
            (extractions_dir / file.name).symlink_to(file)

    # The usual extraction code follows

    os.chdir(extractions_dir)
    flt_files = [str(s) for s in extractions_dir.glob("*GrismFLT.fits")][:]

    grp = multifit.GroupFLT(
        grism_files=flt_files,
        catalog=f"{field_name}-ir.cat.fits",
        cpu_count=-1,
        sci_extn=1,
        pad=800,
    )

    out_dir = Path.cwd()
    #  / conf_type
    # out_dir.mkdir(exist_ok=True, parents=True)
    for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
        (out_dir / filetype).mkdir(exist_ok=True, parents=True)

    max_size = 450
    min_size = 10

    use_idx = []

    for i, row in enumerate(z_cat):

        use_idx.append(i)

        obj_id = row["id_photcat"]
        obj_z = row["zbest"]

        # For some reason, this object segfaults
        if obj_id == 1255:
            continue

        if (out_dir / "full" / f"{field_name}_{obj_id:05}.full.fits").is_file():
            print(f"{obj_id} already extracted.")
            continue

        # Maximum diagonal extent of detection bounding box, measured from centre
        # det_diag = np.sqrt((row["xmax"] - row["XMIN"])**2 + (row["YMAX"] - row["YMIN"])**2)
        det_halfdiag = np.sqrt(
            (np.nanmax([row["xmax"] - row["x"], row["x"] - row["xmin"]])) ** 2
            + (np.nanmax([row["ymax"] - row["y"], row["y"] - row["ymin"]])) ** 2
        )

        # pixel scale is approximately half the detection image
        # Include factor of 25% to account for blotting and pixelation effects
        est_beam_size = int(
            np.nanmin(
                [np.nanmax([np.ceil(0.5 * 1.25 * det_halfdiag), min_size]), max_size]
            )
        )

        # Change parameters here for the drizzled emission line outputs
        pline = {
            "kernel": "square",
            "pixfrac": 1.0,
            "pixscale": 0.06,
            "size": int(np.clip(2 * est_beam_size * 0.06, a_min=3, a_max=30)),
            "wcs": None,
        }
        args = auto_script.generate_fit_params(
            pline=pline,
            field_root=field_name,
            min_sens=0.0,
            min_mask=0.0,
            include_photometry=False,  # set both of these to True to include photometry in fitting
            use_phot_obj=False,
        )

        try:
            if not (
                out_dir / "beams" / f"{field_name}_{obj_id:05}.beams.fits"
            ).is_file():

                print(f"Fetching beams for {obj_id}...")
                beams = grp.get_beams(
                    obj_id,
                    size=est_beam_size,
                    min_mask=0,
                    min_sens=0,
                    beam_id="A",
                )
                mb = multifit.MultiBeam(
                    beams, fcontam=0.2, min_sens=0.0, min_mask=0, group_name=field_name
                )
                # mb.fit_trace_shift()
                # _ = mb.oned_figure()
                #     _ = mb.drizzle_grisms_and_PAs(size=32, scale=0.5, diff=False)
                mb.write_master_fits()

                print(f"Saved beams for {obj_id}.")
            else:
                shutil.copy2(
                    out_dir / "beams" / f"{field_name}_{obj_id:05}.beams.fits",
                    f"{field_name}_{obj_id:05}.beams.fits",
                )
            print(f"Fitting {obj_id}...")

            _ = fitting.run_all_parallel(
                int(obj_id),
                # zr=[obj_z - 0.05, obj_z + 0.05],
                zr=[0.5, 5.2],
                dz=[0.01, 0.0001],
                verbose=True,
                get_output_data=True,
                skip_complete=False,
                save_figures=True,
            )
            print("Fit complete, output saved.")
            for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
                [
                    p.rename(out_dir / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]
        except:
            print(f"Extraction failed for {obj_id}.")
