"""An example workflow for reducing NIRISS/WFSS data from GLASS-JWST ERS."""

import argparse
import os
import tomllib
from pathlib import Path

import numpy as np
from astropy.table import Table

parser = argparse.ArgumentParser(description="Run the full data reduction pipeline.")
parser.add_argument(
    "config_path",
    type=str,
    metavar="config_path",
    help="The path of the configuration file used to setup the fits.",
)
args = parser.parse_args()

with open(args.config_path, "rb") as f:
    config = tomllib.load(f)

# detect if run through mpiexec/mpirun
MPI_avail = False
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    MPI_avail = True

except ImportError:
    print("Could not import MPI")
    mpi_rank = 0
    mpi_size = 1

has_astroclour = False
try:
    import astrocolour

    has_astroclour = True
except:
    pass

print(f"MPI: {mpi_rank=}, {mpi_size=}")

# Latest context
os.environ["CRDS_CONTEXT"] = f"jwst_{config["calibrations"].get("crds_ver", 1535)}.pmap"
os.environ["CRDS_SERVER_URL"] = config["calibrations"].get(
    "crds_server_url", "https://jwst-crds.stsci.edu"
)
os.environ["CRDS_MODE"] = config["calibrations"].get("crds_mode", "auto")
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = config["calibrations"].get(
    "niriss_calib", "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"
)

if mpi_rank == 0:
    # Symlink the custom configuration files.
    # The format for these is pretty self explanatory, and the current set
    # combines the NGDEEP configuration for the first order, with the grizli defaults
    # for all other orders.
    for orig in (Path(__file__).parent / "conf_data").glob("*"):
        if not (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).exists():
            (Path(os.getenv("GRIZLI")) / "CONF" / orig.name).symlink_to(
                orig, target_is_directory=orig.is_dir()
            )
if MPI_avail:
    comm.Barrier()

# https://github.com/PJ-Watson/niriss-tools
from niriss_tools.pipeline import (
    gen_associations,
    process_using_aws,
    queryMAST,
    recursive_merge,
    stsci_det1,
)

root_dir = Path(os.path.expandvars(config["general"].get("root_dir", Path.cwd())))
field = config["general"].get("field")

field_name = f"{config["general"].get("field_prefix")}-{field}".lower()

field_obs_IDs = config["general"].get("field_obs_ids")
proposal_IDs = config["general"].get("proposal_ids")

reduction_ver = config["general"].get("reduction_ver", "1.0.0")

reduction_dir = root_dir / reduction_ver / field_name
reduction_dir.mkdir(exist_ok=True, parents=True)

# Guard against running offline processing without the required files
# in place. Slightly messy since this has to kill all MPI processes.
offline_needed = False
if mpi_rank == 0:
    if (not len(list(reduction_dir.glob(f"{field_name}.*.radec"))) > 0) and (
        config["general"].get("process_offline", False)
    ):
        print(
            "You are attempting to run the processing in offline mode, ",
            "but the required files are not available. Either set",
            "`process_offline = false` under `[general]` in the config",
            "file, or run `A__offline_preprocess.py` first.",
            sep="\n",
        )
        offline_needed = True
        if MPI_avail:
            comm.bcast(offline_needed, root=0)
else:
    offline_needed = comm.bcast(offline_needed, root=0)

if offline_needed:
    exit()

# Cleaner way to point to files than hardcode in config
if config["general"].get("process_offline", False):
    poss_radec = list(reduction_dir.glob(f"{field_name}.*.radec"))
    poss_radec.sort()
    if config["grizli_processing"].get("master_radec") is None:
        config["grizli_processing"]["master_radec"] = str(poss_radec[0])

if __name__ == "__main__":

    level_1_dir = reduction_dir / "Level1"

    if (mpi_rank == 0) and (not config["general"].get("skip_stage_1", True)):

        # Find the correct observations (utils.py is from passagepipe, I couldn't figure out
        # how to access the raw files on MAST until checking that)
        if not (root_dir / f"MAST_summary_{proposal_IDs}.csv").is_file():
            all_obs_tab = queryMAST(proposal_IDs)
            all_obs_tab.write(
                root_dir / f"MAST_summary_{proposal_IDs}.csv", overwrite=True
            )
        else:
            all_obs_tab = Table.read(root_dir / f"MAST_summary_{proposal_IDs}.csv")

        # Any other checks to add here?
        # field_obs_tab = all_obs_tab
        # all_obs_tab.pprint()
        field_obs_tab = all_obs_tab[np.isin(all_obs_tab["obs_id_num"], field_obs_IDs)]

        # print(field_obs_tab)

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

        # Create the _rate.fits files
        stsci_det1(MAST_dir, level_1_dir, **config["level_1"])

    # If this is run using MPI, ensure that the Level 1 detector pipeline
    # finishes before starting the grizli processing
    if MPI_avail:
        comm.Barrier()

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
    prep_dir = grizli_home_dir / "Prep"
    extractions_dir = grizli_home_dir / "Extractions"

    if mpi_rank == 0:
        # Set up the grizli directory structure
        grizli_home_dir.mkdir(exist_ok=True, parents=True)
        prep_dir.mkdir(exist_ok=True)
        (grizli_home_dir / "RAW").mkdir(exist_ok=True)
        (grizli_home_dir / "visits").mkdir(exist_ok=True)
        extractions_dir.mkdir(exist_ok=True)
    if MPI_avail:
        comm.Barrier()

    # As PASSAGE was not ingested into the DJA in the same way as
    # other fields (e.g. GLASS), we have to create an association
    # table ourselves. This contains info on the instrument, filters,
    # footprints, and filenames per group
    assoc_dict = gen_associations(level_1_dir, field_name)

    if not (prep_dir / f"{field_name}-ir_drc_sci.fits").is_file():

        if mpi_rank == 0:
            process_using_aws(
                grizli_home_dir,
                level_1_dir,
                assoc_dict,
                field_name=field_name,
                proposal_id=proposal_IDs,
                process_visit_kwargs=config.get("grizli_processing", {}),
                **config.get("mosaics", {}),
            )

    if MPI_avail:
        comm.Barrier()

    os.chdir(prep_dir)

    if config["subtract_diffuse"].get("run_subtract", False):
        try:
            hdr = fits.getheader(prep_dir / f"{field_name}-ir_drc_sci.fits")
            assert hdr.get("GRBKGSUB", False), "Running grism background subtraction"
        except:
            if mpi_rank == 0:
                from niriss_tools.pipeline import grism_background_subtraction

                grism_background_subtraction(
                    field_root=field_name,
                    grism_prep_kwargs=config["grism_prep"],
                    **config["subtract_diffuse"],
                )
            else:
                pass

    if MPI_avail:
        comm.Barrier()

    if (
        (mpi_rank == 0)
        and (config["general"].get("save_colour_image", False))
        and (not (prep_dir / f"{field_name}_rgb_direct.png").is_file())
        and (has_astroclour)
    ):

        import astropy.visualization as astrovis
        import matplotlib.pyplot as plt
        from astrocolour import ColourImage
        from astrocolour.utils import RawAsinhTransform

        cimg = ColourImage(
            [
                fits.getdata(filepath)
                for filepath in prep_dir.glob(f"{field_name}-f*_drc_sci.fits")
            ],
            transformation_kwargs=dict(
                transformation=RawAsinhTransform(
                    astrovis.ManualInterval(-0.01, 10), 0.05
                )
            ),
        )

        plt.imsave(
            prep_dir / f"{field_name}_rgb_direct.png", cimg, origin="lower", dpi=300
        )
        cimg.adjust_saturation()
        cimg = np.clip(cimg, 0, 1)
        plt.imsave(
            prep_dir / f"{field_name}_rgb_direct_saturation.png",
            cimg,
            origin="lower",
            dpi=300,
        )

    # Require photometric catalogue
    if (not (Path.cwd() / f"{field_name}_phot.fits").is_file()) and (mpi_rank == 0):

        multiband_catalog_args = auto_script.get_yml_parameters()[
            "multiband_catalog_args"
        ]

        catalog_kwargs = recursive_merge(
            multiband_catalog_args, config["multiband_catalogue"]
        )

        # Require detection catalogue first
        if (not (Path.cwd() / f"{field_name}-ir.cat.fits").is_file()) & (
            config["multiband_catalogue"].get("force_seg_map", None) is not None
        ):

            from astropy.wcs import WCS
            from niriss_tools.isophotal import reproject_image
            from niriss_tools.pipeline import regen_catalogue

            # The point of this is to align to the v0.5 reduction seg map
            old_seg_name = Path(
                config["multiband_catalogue"].get("force_seg_map", None)
            )

            aligned_seg_name = prep_dir / f"aligned_{old_seg_name.name}"

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

            catalog_kwargs["run_detection"] = False
            catalog_kwargs.pop("force_seg_map")

            new_cat = regen_catalogue(
                use_regen_seg, root=f"{field_name}-ir", **catalog_kwargs
            )

        phot_cat = auto_script.multiband_catalog(
            field_root=field_name,
            **catalog_kwargs,
        )

    if MPI_avail:
        comm.Barrier()

    os.chdir(prep_dir)

    rate_files = [str(s) for s in Path.cwd().glob("*_rate.fits")][:]
    grism_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    if (len(grism_files) == 0) and (mpi_rank == 0):

        grism_prep_kwargs = auto_script.get_yml_parameters()["grism_prep_args"]
        kwargs = recursive_merge(grism_prep_kwargs, config["grism_prep"])

        if config["general"].get("low_memory", True):

            visits, groups, info = auto_script.load_visits_yaml(
                Path.cwd() / f"{field_name}_visits.yaml"
            )

            for pupil in np.unique(info["PUPIL"]):
                os.chdir(prep_dir)
                pupil_rate_files = [
                    str(Path.cwd() / f) for f in info[info["PUPIL"] == pupil]["FILE"]
                ]
                kwargs["files"] = pupil_rate_files[:]

                grp = auto_script.grism_prep(field_root=field_name, **kwargs)
        else:

            grism_prep_kwargs["files"] = rate_files[:]

            grp = auto_script.grism_prep(field_root=field_name, **kwargs)

    # exit()

    # Ensure that all processed files are correctly linked to the
    # Extractions directory
    if mpi_rank == 0:
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

    if MPI_avail:
        comm.Barrier()

    # exit()

    # The usual extraction code follows

    os.chdir(extractions_dir)

    flt_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    grp = multifit.GroupFLT(
        grism_files=flt_files,
        catalog=f"{field_name}-ir.cat.fits",
        cpu_count=config["extraction"].get("cpu_count", 4),
        sci_extn=1,
        pad=config["grism_prep"].get("pad", 800),
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
        # bad_pa_threshold=10,
        # diff2d=2,
    )

    # Some examples
    galaxies = {
        # LCS
        # 614: 3.1,
        # 1615: 1.94,
        # 1516: 2.2,
        # 1633: 1.97,
        # 1855: 2.8,
        # # 3028: 0.87
        611: 3.1,
    }

    for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
        (extractions_dir / filetype).mkdir(exist_ok=True, parents=True)

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
                # zr=[obj_z - 0.05, obj_z + 0.05],
                zr=[obj_z - 0.2, obj_z + 0.2],
                # zr=[0, 5.2],
                dz=[0.001, 0.0001],
                verbose=True,
                get_output_data=True,
                skip_complete=False,
                save_figures=True,
            )

            for filetype in ["beams", "full", "1D", "row", "line", "log_par", "stack"]:
                [
                    p.rename(extractions_dir / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]
