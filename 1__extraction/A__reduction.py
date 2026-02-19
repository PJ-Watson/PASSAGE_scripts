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


def grism_prep_wrapper(
    field_name: str = field_name,
    rate_files: list = [],
    grism_prep_kwargs: dict = {},
    cpu_count: int = 1,
    flt_pad: int = 800,
):
    """
    A simple wrapper to ensure consistent processing of the dispersed spectra.

    Parameters
    ----------
    field_name : str, optional
        The name of the field, by default "glass-a2744".
    rate_files : list, optional
        A list of `*_rate.fits` files, by default [].
    grism_prep_kwargs : dict, optional
        A dictionary of parameters for `auto_script.grism_prep`, by default {}.
    cpu_count : int, optional
        The number of cores to use for processing, by default 1.
    flt_pad : int, optional
        The padding to add to the edges of the images, by default 800.
    """

    # For now, turn off refining contamination model with polynomial fits
    grism_prep_kwargs["refine_niter"] = 0
    # grism_prep_args["refine_poly_order"] = 7

    # Flat-flambda spectra
    grism_prep_kwargs["init_coeffs"] = [1.0]

    grism_prep_kwargs["mask_mosaic_edges"] = False

    # Here we use all of the detected objects.
    # These can be adjusted based on how deep the spectra/visits are
    grism_prep_kwargs["refine_mag_limits"] = [14.0, 50.0]
    grism_prep_kwargs["prelim_mag_limit"] = 50.0

    # The grism reference filters for direct images
    grism_prep_kwargs["gris_ref_filters"] = {
        "GR150R": ["F115W", "F150W", "F200W"],
        "GR150C": ["F115W", "F150W", "F200W"],
    }

    grism_prep_kwargs["use_jwst_crds"] = False
    grism_prep_kwargs["files"] = rate_files[:]

    grism_prep_kwargs["model_kwargs"] = {
        "compute_size": True,
    }

    grp = auto_script.grism_prep(
        field_root=field_name, pad=flt_pad, cpu_count=cpu_count, **grism_prep_kwargs
    )


if __name__ == "__main__":

    # Find the correct observations (utils.py is from passagepipe, I couldn't figure out
    # how to access the raw files on MAST until checking that)
    if not (root_dir / f"MAST_summary_{proposal_ID}.csv").is_file():
        all_obs_tab = utils.queryMAST(proposal_ID)
        all_obs_tab.write(root_dir / f"MAST_summary_{proposal_ID}.csv", overwrite=True)
    else:
        all_obs_tab = Table.read(root_dir / f"MAST_summary_{proposal_ID}.csv")

    # Any other checks to add here?
    # field_obs_tab = all_obs_tab
    field_obs_tab = all_obs_tab[np.isin(all_obs_tab["obs_id_num"], field_obs_IDs)]

    print(field_obs_tab)

    from mastquery import utils as mastutils

    MAST_dir = reduction_dir / "MAST_downloads"
    MAST_dir.mkdir(exist_ok=True, parents=True)

    level_1_dir = reduction_dir / "Level1"
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

    # Create the _rate.fits files
    pipeline.stsci_det1(MAST_dir, level_1_dir, cpu_count=2)

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

    grizli_home_dir.mkdir(exist_ok=True, parents=True)
    (grizli_home_dir / "Prep").mkdir(exist_ok=True)
    (grizli_home_dir / "RAW").mkdir(exist_ok=True)
    (grizli_home_dir / "visits").mkdir(exist_ok=True)

    # As PASSAGE was not ingested into the DJA in the same way as
    # other fields (e.g. GLASS), we have to create an association
    # table ourselves. This contains info on the instrument, filters,
    # footprints, and filenames per group
    assoc_dict = pipeline.gen_associations(level_1_dir, field_name)

    if not (grizli_home_dir / "Prep" / f"{field_name}-ir_drc_sci.fits").is_file():

        # assoc_dict = pipeline.load_assoc()

        pipeline.process_using_aws(
            grizli_home_dir, level_1_dir, assoc_dict, field_name=field_name
        )

    # Set up the grizli extraction directory structure
    (grizli_home_dir / "Extractions").mkdir(exist_ok=True)

    os.chdir(grizli_home_dir / "Prep")

    # Subtract diffuse background component (e.g. ICL in GLASS)
    # Not necessary for PASSAGE, can cause problems around bright stars
    # try:
    #     hdr = fits.getheader(grizli_home_dir / "Prep" / f"{field_name}-ir_drc_sci.fits")
    #     assert hdr.get("GRBKGSUB", False), "Running grism background subtraction"
    # except:
    #     pipeline.grism_background_subtraction(field_root=field_name, grism_prep_fn=grism_prep_wrapper)

    if not (Path.cwd() / f"{field_name}_phot.fits").is_file():

        if not (Path.cwd() / f"{field_name}-ir.cat.fits").is_file():

            from astropy.wcs import WCS
            from niriss_tools.isophotal import reproject_image
            from niriss_tools.pipeline import regen_catalogue

            # The point of this is to align to the v0.5 reduction seg map
            # old_seg_name = passage_dir / f"{field_name.capitalize()}_det_drz_seg.fits"
            old_seg_name = list(reduction_dir.glob("*det_drz_seg.fits"))[0]
            # (Or whatever name you came up with during the previous reduction)

            aligned_seg_name = grizli_home_dir / "Prep" / f"aligned_{old_seg_name.name}"

            reproject_image(
                old_seg_name,
                aligned_seg_name,
                WCS(fits.getheader(f"{field_name}-ir_drc_sci.fits")),
                fits.getdata(f"{field_name}-ir_drc_sci.fits").shape,
                method="interp",
                order="nearest-neighbor",
            )

            segment_map = fits.getdata(aligned_seg_name)

            use_regen_seg = np.asarray(segment_map).astype(np.int32)

            new_cat = regen_catalogue(
                use_regen_seg,
                root=f"{field_name}-ir",
            )

        exist_cat_name = f"{field_name}-ir.cat.fits"

        multiband_catalog_args = auto_script.get_yml_parameters()[
            "multiband_catalog_args"
        ]
        multiband_catalog_args["run_detection"] = False
        multiband_catalog_args["filters"] = [
            "f115wn-clear",
            "f150wn-clear",
            "f200wn-clear",
        ]

        phot_cat = auto_script.multiband_catalog(
            field_root=field_name,
            # master_catalog=exist_cat_name,
            **multiband_catalog_args,
        )

    # exit()

    kwargs = auto_script.get_yml_parameters()

    # The number of processes to use
    cpu_count = 4

    os.chdir(grizli_home_dir / "Prep")

    rate_files = [str(s) for s in Path.cwd().glob("*_rate.fits")][:]
    grism_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    if len(grism_files) == 0:

        grism_prep_wrapper(
            rate_files=rate_files, grism_prep_kwargs=kwargs["grism_prep_args"]
        )

    # The usual extraction code follows

    os.chdir(grizli_home_dir / "Extractions")

    # # Remove bad exposure in GLASS (should have done this earlier tbh)
    # for f in (grizli_home_dir / "Extractions").glob("jw01324001001_09101_00002_nis*"):
    #     if f.is_symlink():
    #         f.unlink()

    flt_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    grp = multifit.GroupFLT(
        grism_files=flt_files,
        catalog=f"{field_name}-ir.cat.fits",
        cpu_count=-1,
        sci_extn=1,
        pad=800,
    )

    pline = {
        "kernel": "square",
        "pixfrac": 1.0,
        "pixscale": 0.06,
        "size": 5,
        "wcs": None,
    }
    args = auto_script.generate_fit_params(
        pline=pline,
        field_root=field_name,
        min_sens=0.0,
        min_mask=0.0,
        # Set both of these to True to include photometry in fitting
        include_photometry=False,
        use_phot_obj=False,
    )

    # Some examples
    galaxies = {
        578: 1.438,
        1779: 1.372,
        1997: 1.406,
        2056: 1.865,
        2275: 3.079,
        2993: 1.248,
        243: 1.247,
        721: 1.265,
        762: 2.277,
        1275: 7.925,
    }

    for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
        (grizli_home_dir / "Extractions" / filetype).mkdir(exist_ok=True, parents=True)

    for obj_id, obj_z in galaxies.items():

        if not (
            grizli_home_dir
            / "Extractions"
            / "full"
            / f"{field_name}_{obj_id:0>5}.full.fits"
        ).is_file():

            beams = grp.get_beams(
                int(obj_id),
                size=50,
                min_mask=0,
                min_sens=0,
                show_exception=True,
                beam_id="A",
            )
            mb = multifit.MultiBeam(
                beams, fcontam=0.2, min_sens=0.0, min_mask=0, group_name=field_name
            )

            # This produces unusual offsets in the emission line maps.
            # Probably a bug in grizli that I don't have the energy to
            # chase down anymore.
            # mb.fit_trace_shift()
            # 2025-12-06: Should be fixed in my fork, but needs more testing

            mb.write_master_fits()

            _ = fitting.run_all_parallel(
                int(obj_id),
                zr=[obj_z - 0.05, obj_z + 0.05],
                # zr=[0, 12],
                dz=[0.001, 0.0001],
                verbose=True,
                get_output_data=True,
                skip_complete=False,
                save_figures=True,
            )

            for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
                [
                    p.rename(grizli_home_dir / "Extractions" / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]
