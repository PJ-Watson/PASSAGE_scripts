"""An example workflow for reducing NIRISS/WFSS data from GLASS-JWST ERS."""

import os
import shutil
from pathlib import Path

import numpy as np

# https://github.com/PJ-Watson/pygrife
import pygrife
from astropy.table import Table

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1467.pmap"
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"

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

# Merge some objects
new_id_mapping = {
    10930: [930, 931],
    10966: [966, 967],
    11118: [1118, 1119],
    11368: [1368, 1369],
    11674: [1674, 1675],
    13154: [3154, 3155, 3156],
    13356: [3356, 3357],
    10713: [713, 714],
    243: [243],
    721: [721],
    762: [762],
}

# Estimated redshifts for the merged objects
new_id_redshifts = {
    10930: 2.275,
    10966: 1.825,
    11118: 2.273,
    11368: 1.260,
    11674: 1.431,
    13154: 1.70,
    13356: 3.08,
    10713: 0.891,
    243: 1.247,
    721: 1.265,
    762: 2.277,
}


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
    prep_dir = grizli_home_dir / f"Prep"
    extractions_dir = grizli_home_dir / f"Extractions"

    forced_dir = grizli_home_dir / f"ForcedExtractions"
    forced_dir.mkdir(exist_ok=True, parents=True)

    default_seg_path = extractions_dir / f"{field_name}-ir_seg.fits"

    forced_seg_path = forced_dir / f"{field_name}-ir_seg_forced.fits"
    if not forced_seg_path.is_file():

        with fits.open(default_seg_path) as default_hdul:
            seg_data = default_hdul[0].data

            for new_id, old_ids in new_id_mapping.items():
                seg_data[np.where(np.isin(seg_data, old_ids))] = new_id
                # print (new_id, "prev")

            default_hdul[0].data = seg_data
            default_hdul.writeto(forced_seg_path)
    # exit()

    old_seg = fits.getdata(default_seg_path)
    forced_seg = fits.getdata(forced_seg_path)

    os.chdir(forced_dir)
    os.system(f"ln -s ../Prep/*GrismFLT* .")
    os.system(f"ln -s ../Prep/*.0?.wcs.fits .")
    os.system(f"ln -s ../Prep/*dr[zc]*.fits .")

    if not (forced_dir / f"{field_name}-ir_seg.fits").is_file():
        from niriss_tools.pipeline import regen_catalogue

        segment_map = fits.getdata(forced_seg_path)

        os.chdir(forced_dir)

        use_regen_seg = np.asarray(segment_map).astype(np.int32)
        new_cat = regen_catalogue(
            use_regen_seg,
            root=f"{field_name}-ir",
        )

    # Set up the forced extraction tool
    ge = pygrife.GrismExtractor(
        field_root=field_name,
        in_dir=prep_dir,
        out_dir=forced_dir,
        seg_path=forced_dir / f"{field_name}-ir_seg.fits",
    )
    ge.load_grism_files(
        cpu_count=4,
        # use_jwst_crds=True if "STScI" in conf_type else False,
        use_jwst_crds=False,
    )

    os.chdir(forced_dir)

    for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
        (ge.out_dir / filetype).mkdir(exist_ok=True, parents=True)

    max_size = 450
    min_size = 10

    forced_cat = Table.read(f"{field_name}-ir.cat.fits")

    for i, row in enumerate(forced_cat):

        obj_id = row["NUMBER"]

        if obj_id not in new_id_mapping.keys():
            continue

        # obj_z = z_cat[new_id_mapping[obj_id][0] == z_cat["ID_NIRISS"]]["Z_NIRISS"]
        # obj_z = fits.getheader(extractions_dir / "full" / f"{field_name}_{new_id_mapping[obj_id][0]:0>5}.full.fits")["REDSHIFT"]
        obj_z = new_id_redshifts[obj_id]

        print(obj_id, obj_z)

        if np.nansum(forced_seg == obj_id) == np.nansum(old_seg == obj_id):
            continue

        if (ge.out_dir / "full" / f"{field_name}_{obj_id:05}.full.fits").is_file():
            print(f"{obj_id} already extracted.")
            continue

        # Maximum diagonal extent of detection bounding box, measured from centre
        # det_diag = np.sqrt((row["XMAX"] - row["XMIN"])**2 + (row["YMAX"] - row["YMIN"])**2)
        det_halfdiag = np.sqrt(
            (np.nanmax([row["XMAX"] - row["X"], row["X"] - row["XMIN"]])) ** 2
            + (np.nanmax([row["YMAX"] - row["Y"], row["Y"] - row["YMIN"]])) ** 2
        )

        est_beam_size = int(
            np.nanmin(
                [np.nanmax([np.ceil(0.5 * 1.25 * det_halfdiag), min_size]), max_size]
            )
        )

        pline = {
            "kernel": "square",
            "pixfrac": 1.0,
            "pixscale": 0.06,
            "size": int(np.clip(2 * est_beam_size * 0.06, a_min=3, a_max=30)),
            "wcs": None,
        }
        beams_kwargs = {
            "size": est_beam_size,
            "min_sens": 0.0,
            "min_mask": 0.0,
            "beam_id": "A",
        }
        multibeam_kwargs = {
            "fcontam": 0.2,
            "min_sens": 0.0,
            "min_mask": 0.0,
            # "group_name": root_name,
        }
        fit_kwargs = {
            "pline": pline,
            # "field_root" : root_name,
            "min_sens": 0.0,
            "min_mask": 0.0,
            "dz": [0.01, 0.001],
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

            print(f"Fetching beams for {obj_id}...")
            ge.extract_spectra(
                obj_id,
                fit_kwargs=fit_kwargs,
                multibeam_kwargs=multibeam_kwargs,
                beams_kwargs=beams_kwargs,
                z_range=[obj_z - 0.05, obj_z + 0.05],
            )
            for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
                [
                    p.rename(ge.out_dir / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]

            # shutil.copy2(ge.out_dir / f"{field_name}_{obj_id:05}.beams.fits", out_dir / "beams" / f"{field_name}_{obj_id:05}.beams.fits")
            # else:
            shutil.copy2(
                ge.out_dir / "beams" / f"{field_name}_{obj_id:05}.beams.fits",
                f"{field_name}_{obj_id:05}.beams.fits",
            )
            print(f"Fitting {obj_id}...")

            _ = fitting.run_all_parallel(
                int(obj_id),
                zr=[obj_z - 0.05, obj_z + 0.05],
                dz=[0.001, 0.0001],
                verbose=True,
                get_output_data=True,
                skip_complete=False,
                save_figures=True,
            )
            print("Fit complete, output saved.")
            for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
                [
                    p.rename(extractions_dir / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]
        except:
            print(f"Extraction failed for {obj_id}.")
