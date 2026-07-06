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

parser = argparse.ArgumentParser(description="Run the data extraction pipeline.")
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

print(f"MPI: {mpi_rank=}, {mpi_size=}")

# Latest context
os.environ["CRDS_CONTEXT"] = f"jwst_{config["calibrations"].get("crds_ver", 1535)}.pmap"
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
from niriss_tools.grism.utils import gen_stacked_beams
from niriss_tools.pipeline import separate_oned_spectra

root_dir = Path(os.path.expandvars(config["general"].get("root_dir", Path.cwd())))
field = config["general"].get("field")

field_name = f"{config["general"].get("field_prefix")}-{field}".lower()

field_obs_IDs = config["general"].get("field_obs_ids")
proposal_IDs = config["general"].get("proposal_ids")

reduction_ver = config["general"].get("reduction_ver", "1.0.0")

reduction_dir = root_dir / reduction_ver / field_name
reduction_dir.mkdir(exist_ok=True, parents=True)

import warnings

# Quiet some of the common the grizli-induced warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astropy.wcs import FITSFixedWarning

for w in [VerifyWarning, FITSFixedWarning, UnitsWarning, RuntimeWarning]:
    warnings.simplefilter("ignore", category=w)

# In case some objects need to be filtered out manually
bad_objs = []

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
    prep_dir = grizli_home_dir / "Prep"
    extractions_dir = grizli_home_dir / "Extractions"

    if mpi_rank == 0:
        prep_dir.mkdir(exist_ok=True, parents=True)
        extractions_dir.mkdir(exist_ok=True, parents=True)

    kwargs = auto_script.get_yml_parameters()

    # The number of files to load on each process - not used in this script
    # chunk_size = config["extraction"].get("chunk_size", 8)
    # cpu_count = 8

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
    if mpi_rank == 0:
        for file in files_to_link:
            if not (extractions_dir / file.name).is_file():
                (extractions_dir / file.name).symlink_to(file)
    if MPI_avail:
        comm.Barrier()

    # The usual extraction code follows

    os.chdir(extractions_dir)

    if mpi_rank == 0:
        flt_files = [str(s) for s in extractions_dir.glob("*GrismFLT.fits")][:]
        flt_files.sort()
        grism_files_split = np.array_split(flt_files, mpi_size)
    else:
        grism_files_split = None

    grism_files_split = comm.scatter(grism_files_split, root=0)

    filetype_list = [
        "beams",
        "beams_stacked",
        "full",
        "1D_RC",
        "1D",
        "row",
        "line",
        "log_par",
        "tfit",
        "stack",
    ]

    if mpi_rank == 0:
        for filetype in filetype_list:
            (extractions_dir / filetype).mkdir(exist_ok=True, parents=True)
            # for chunk_i, grism_subset in enumerate(np.array_split(flt_files, mpi_size)):
            for i in np.arange(mpi_size):
                (extractions_dir / "beams" / f"process_{i}_{mpi_size}").mkdir(
                    exist_ok=True, parents=True
                )
    if MPI_avail:
        comm.Barrier()

    max_size = config["extraction"].get("max_size", 150)
    min_size = config["extraction"].get("min_size", 15)

    mag_limit = config["extraction"].get("mag_limit", 50)

    if mpi_rank == 0:
        phot_cat = Table.read(extractions_dir / f"{field_name}_phot.fits")

        # phot_cat = phot_cat[np.isin(phot_cat["id"], candidate_obj_ids)]
        phot_cat = phot_cat[phot_cat["mag_auto"] < mag_limit]

        if config["extraction"].get("recalculate_size"):

            # The extra factor is to account for the different pixel scale
            # between the mosaic and the grism data, and a small fudge in case
            # of blotting effects or pixelation artefacts
            phot_cat["est_extent"] = (
                np.sqrt(2)
                * np.nanmax(
                    [
                        phot_cat["xmax"] - phot_cat["x"],
                        phot_cat["x"] - phot_cat["xmin"],
                        phot_cat["ymax"] - phot_cat["y"],
                        phot_cat["y"] - phot_cat["ymin"],
                    ],
                    axis=0,
                )
                * 0.5
                * 1.2
            )
            phot_cat["beam_size"] = np.nanmin(
                [
                    np.nanmax(
                        [
                            phot_cat["est_extent"],
                            np.full_like(phot_cat["est_extent"], min_size),
                        ],
                        axis=0,
                    ),
                    np.full_like(phot_cat["est_extent"], max_size),
                ],
                axis=0,
            ).astype(int)
        else:
            phot_cat["beam_size"] = np.full_like(
                phot_cat["x"], config["extraction"].get("beam_size", 50)
            ).astype(int)
        phot_cat.pprint()

        # Check the status of all objects in the catalogue
        phot_cat["status"] = 0
        for i, row in enumerate(phot_cat):
            obj_id = row["id"]

            if (
                extractions_dir / "full" / f"{field_name}_{obj_id:0>5}.full.fits"
            ).is_file():
                phot_cat["status"][i] = 0
            elif (
                extractions_dir
                / "beams_stacked"
                / f"{field_name}_{obj_id:0>5}.beams.fits"
            ).is_file():
                phot_cat["status"][i] = 1
            elif (
                extractions_dir / "beams" / f"{field_name}_{obj_id:0>5}.beams.fits"
            ).is_file():
                phot_cat["status"][i] = 2
            else:
                phot_cat["status"][i] = 3

            if phot_cat["status"][i] < 3:
                for m in (extractions_dir / "beams").glob(
                    f"*/{field_name}_{obj_id:0>5}.beams.fits"
                ):
                    m.unlink()
    else:
        phot_cat = None
    if MPI_avail:
        phot_cat = comm.bcast(phot_cat, root=0)

    beams_cat = phot_cat.copy()
    beams_cat = beams_cat[beams_cat["status"] == 3]
    process_dir = extractions_dir / "beams" / f"process_{mpi_rank}_{mpi_size}"
    beams_cat["beams_status"] = [
        (process_dir / f"{field_name}_{obj_id:0>5}.beams.fits").is_file()
        for obj_id in beams_cat["id"]
    ]
    beams_cat = beams_cat[np.logical_not(beams_cat["beams_status"])]

    if len(beams_cat) > 0:

        # for chunk_i, grism_subset in enumerate(
        #     np.split(
        #         grism_files_split,
        #         np.arange(chunk_size, len(grism_files_split), chunk_size),
        #     )
        # ):

        os.chdir(extractions_dir)

        # sub_beams_dir = process_dir / f"sub_{chunk_i}"
        # sub_beams_dir.mkdir(exist_ok=True)

        beams_extracted = [
            (process_dir / f"{field_name}_{obj_id:0>5}.beams.fits").is_file()
            for obj_id in beams_cat["id"]
        ]

        if not all(beams_extracted):

            grp = multifit.GroupFLT(
                grism_files=grism_files_split,
                catalog=f"{field_name}-ir.cat.fits",
                cpu_count=config["extraction"].get("cpu_count", 1),
                sci_extn=1,
                pad=config["grism_prep"].get("pad", 800),
            )
            os.chdir(process_dir)

            for i, row in enumerate(beams_cat[np.logical_not(beams_extracted)]):

                try:
                    beam_kwargs = config["extraction"].get("beams", {})
                    beams = grp.get_beams(
                        row["id"],
                        size=row["beam_size"],
                        min_mask=beam_kwargs.get("min_mask", 0.0),
                        min_sens=beam_kwargs.get("min_sens", 0.0),
                        min_overlap=beam_kwargs.get("min_overlap", 0.0),
                        beam_id="A",
                    )
                    mb = multifit.MultiBeam(beams, group_name=field_name, **beam_kwargs)
                    if config["extraction"].get("fit_trace_shift", False):
                        mb.fit_trace_shift()
                    # _ = mb.oned_figure()
                    #     _ = mb.drizzle_grisms_and_PAs(size=32, scale=0.5, diff=False)
                    mb.write_master_fits()
                except Exception as e:
                    print(e)
                    pass

        # del grp

        # os.chdir(process_dir)

        # # Merge the chunked beams files
        # for row in beams_cat:
        #     try:
        #         mb_parts_list = [
        #             str(m)
        #             for m in (process_dir).glob(
        #                 f"*/{field_name}_{row["id"]:0>5}.beams.fits"
        #             )
        #         ]
        #         mb = multifit.MultiBeam(
        #             # str(extractions_dir / "backup" /"beams" / f"{field_name}_{obj_id:0>5}.beams.fits"),
        #             mb_parts_list[:],
        #             fcontam=0.2,
        #             min_sens=0.0,
        #             min_mask=0,
        #             group_name=field_name,
        #         )
        #         mb.write_master_fits()
        #     except:
        #         print(f"{mpi_rank=}: No beams found for object {row["id"]:0>5}")

        # # Delete subdirectories when no longer needed
        # for subdir in process_dir.glob("sub_*"):
        #     shutil.rmtree(subdir)

    if mpi_rank == 0:
        # fit_cat = phot_cat[phot_cat["status"]>0]
        idx_arr = np.array_split(np.nonzero(phot_cat["status"] > 0)[0], mpi_size)
        # print (idx_arr)
        # print (np.array_split(idx_arr, 9))
    else:
        idx_arr = None
    if MPI_avail:
        idx_arr = comm.scatter(idx_arr, root=0)
    fit_cat = phot_cat[idx_arr]
    fit_cat.sort(["mag_auto"])

    # print (mpi_rank, phot_cat[idx_arr])

    # exit()

    beam_kwargs = config["extraction"].get("beams", {})

    if mpi_rank == 0:
        os.chdir(extractions_dir)
        args = auto_script.generate_fit_params(
            field_root=field_name,
            **beam_kwargs,
            **config["extraction"].get("fit_params", {}),
        )
    else:
        args = None

    if MPI_avail:
        args = comm.bcast(args, root=0)
        comm.Barrier()

    for i, row in enumerate(fit_cat[:]):

        t0 = time()

        obj_id = row["id"]

        print(f"{mpi_rank=}, {obj_id=}")

        if obj_id in bad_objs:
            continue

        try:

            os.chdir(extractions_dir)

            try:
                mb = multifit.MultiBeam(
                    str(
                        extractions_dir
                        / "beams"
                        / f"{field_name}_{obj_id:0>5}.beams.fits"
                    ),
                    group_name=field_name,
                    **beam_kwargs,
                )
            except:
                mb_parts_list = [
                    str(m)
                    for m in (extractions_dir / "beams").glob(
                        f"*/{field_name}_{obj_id:0>5}.beams.fits"
                    )
                ]
                mb = multifit.MultiBeam(
                    mb_parts_list[:], group_name=field_name, **beam_kwargs
                )
                for m in mb_parts_list:
                    Path(m).unlink()

            # os.chdir(extractions_dir / "beams")
            mb.write_master_fits()
            # os.chdir(extractions_dir)
            if not (
                extractions_dir / "beams" / f"{field_name}_{obj_id:0>5}.beams.fits"
            ).is_file():
                shutil.copy2(
                    Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits",
                    extractions_dir / "beams" / f"{field_name}_{obj_id:0>5}.beams.fits",
                )

            if config["extraction"].get("stack_beams", False):
                try:
                    shutil.copy2(
                        extractions_dir
                        / "beams_stacked"
                        / f"{field_name}_{obj_id:0>5}.beams.fits",
                        f"{field_name}_{obj_id:05}.beams.fits",
                    )
                except:
                    # Cluster and stack the individual beams before fitting
                    new_mb = gen_stacked_beams(
                        mb,
                        group_name=field_name,
                        **config["extraction"].get("stack_kwargs", {}),
                        **beam_kwargs,
                    )

                    # Save and copy immediately to the stacked folder
                    # Avoids rerunning clustering code if things crash
                    # during redshift fitting
                    new_mb.write_master_fits()
                    shutil.copy2(
                        Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits",
                        extractions_dir
                        / "beams_stacked"
                        / f"{field_name}_{obj_id:0>5}.beams.fits",
                    )
                    del new_mb

            del mb

            # Change parameters here for the drizzled emission line outputs
            pline = args.get("pline", {})

            if config["extraction"].get("recalculate_size", True):
                pline["size"] = int(
                    np.clip(2 * row["beam_size"] * 0.06, a_min=3, a_max=30)
                )

            mb, st, fit, tfit, line_hdu = fitting.run_all_parallel(
                int(obj_id),
                pline=pline,
                get_output_data=True,
            )

            try:
                with open(
                    Path.cwd() / f"{field_name}_{obj_id:0>5}.tfit.pickle", "wb"
                ) as pickle_filepath:
                    pickle.dump(tfit, pickle_filepath)
            except Exception as e:
                print(
                    f"{mpi_rank=}: {obj_id=} Failed to write tfit to file: {e}",
                    flush=True,
                )

            try:
                oned_RC = separate_oned_spectra(mb, tfit)
                oned_RC.writeto(Path.cwd() / f"{field_name}_{mb.id:0>5}.1D_RC.fits")
                del oned_RC
                del mb
                del tfit
            except Exception as e:
                print(
                    f"{mpi_rank=}: {obj_id=} Failed to write 1D_RC spectra: {e}",
                    flush=True,
                )

            if config["extraction"].get("stack_beams", False):
                (Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits").rename(
                    extractions_dir
                    / "beams_stacked"
                    / f"{field_name}_{obj_id:0>5}.beams.fits"
                )

            for filetype in filetype_list:
                [
                    p.rename(extractions_dir / filetype / p.name)
                    for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
                ]

            print(f"{mpi_rank=}: {obj_id=} Fit complete, output saved.", flush=True)
            print(f"{mpi_rank=}: {obj_id=} Time taken: {time()-t0}", flush=True)
        except Exception as e:
            print(f"{mpi_rank=}: Fitting failed for {obj_id}: {e}")

    if config.get("line_finding", {}).get("zip_outputs", False):

        from niriss_tools.pipeline.utils import gen_linefinding_outputs

        line_finding_kwargs = config.get("line_finding", {})
        line_finding_kwargs.pop("zip_outputs")
        if "new_field_name" not in line_finding_kwargs:
            line_finding_kwargs["new_field_name"] = field_name[-6:]

        gen_linefinding_outputs(
            grizli_home_dir=grizli_home_dir,
            field_name=field_name,
            **line_finding_kwargs,
        )
