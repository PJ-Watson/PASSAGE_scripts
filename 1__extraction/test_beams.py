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
from niriss_tools.grism import gen_stacked_beams

root_dir = Path(config["general"].get("root_dir", Path.cwd()))
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
bad_objs = [1197, 1199, 1169, 1201, 1200]

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
    extractions_dir = grizli_home_dir / "Extractions-no-fit"

    kwargs = auto_script.get_yml_parameters()

    os.chdir(extractions_dir)

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
    phot_cat = comm.bcast(phot_cat, root=0)

    if mpi_rank == 0:
        # fit_cat = phot_cat[phot_cat["status"]>0]
        idx_arr = np.array_split(np.nonzero(phot_cat["status"] > 0)[0], mpi_size)
        # print (idx_arr)
        # print (np.array_split(idx_arr, 9))
    else:
        idx_arr = None
    idx_arr = comm.scatter(idx_arr, root=0)
    fit_cat = phot_cat[idx_arr]
    fit_cat.sort(["mag_auto"])

    t0 = time()

    if mpi_rank == 0:
        os.chdir(extractions_dir)
        args = auto_script.generate_fit_params(
            field_root=field_name, **config["extraction"].get("beams", {}), **config["extraction"].get("fit_params", {})
        )
    else:
        args = None

    if MPI_avail:
        args = comm.bcast(args, root=0)
        comm.Barrier()

    beam_kwargs = config["extraction"].get("beams", {})
    print (f"{beam_kwargs=}")
    # exit()


    for i, row in enumerate(fit_cat[:]):

        obj_id = row["id"]

        print(f"{mpi_rank=}, {obj_id=}")

        # if obj_id in bad_objs:
        #     continue

        if obj_id != 1169:
            continue

        # try:

        os.chdir(extractions_dir)

        from grizli.multifit import MultiBeam
        mb = MultiBeam(
            str(extractions_dir
            # / ""
            / f"{field_name}_{obj_id:0>5}.beams.fits"),
            group_name=field_name,
            # **beam_kwargs
        )
        bad_pa_threshold=1.6

        print ("Loaded multibeam", flush=True)

        print (np.nansum(mb.fit_mask), flush=True)

        import matplotlib.pyplot as plt
        for beam in mb.beams:
            fig, axs = plt.subplots()
            axs.imshow(beam.fit_mask.reshape(beam.sh))
            plt.show()

        out = mb.check_for_bad_PAs(
            chi2_threshold=bad_pa_threshold,
            poly_order=1,
            reinit=True,
            fit_background=True,
        )

        print (out, flush=True)
        print ("CHecked PAs", flush=True)

        exit()

            # # try:
            # #     shutil.copy2(
            # #         extractions_dir
            # #         / "beams_stacked"
            # #         / f"{field_name}_{obj_id:0>5}.beams.fits",
            # #         f"{field_name}_{obj_id:05}.beams.fits",
            # #     )
            # # except:
            # try:
            #     mb = multifit.MultiBeam(
            #         str(
            #             extractions_dir
            #             / "beams"
            #             / f"{field_name}_{obj_id:0>5}.beams.fits"
            #         ),
            #         group_name=field_name,
            #         **beam_kwargs,
            #     )
            # except:
            #     mb_parts_list = [
            #         str(m)
            #         for m in (extractions_dir / "beams").glob(
            #             f"*/{field_name}_{obj_id:0>5}.beams.fits"
            #         )
            #     ]
            #     mb = multifit.MultiBeam(
            #         mb_parts_list[:], group_name=field_name, **beam_kwargs
            #     )
            #     for m in mb_parts_list:
            #         Path(m).unlink()

            # mb.write_master_fits()
            # shutil.copy2(
            #     Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits",
            #     extractions_dir / "beams" / f"{field_name}_{obj_id:0>5}.beams.fits",
            # )

            # # Cluster and stack the individual beams before fitting
            # new_mb = gen_stacked_beams(
            #     mb,
            #     fcontam=0.2,
            #     min_sens=0.0,
            #     min_mask=0,
            #     group_name=field_name,
            #     cluster_beams=True,
            # )

            # # Save and copy immediately to the stacked folder
            # # Avoids rerunning clustering code if things crash
            # # during redshift fitting
            # new_mb.write_master_fits()
            # shutil.copy2(
            #     Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits",
            #     extractions_dir
            #     / "beams_stacked"
            #     / f"{field_name}_{obj_id:0>5}.beams.fits",
            # )
            # del mb
            # del new_mb

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

        #     # try:

        #     with open(
        #         Path.cwd() / f"{field_name}_{obj_id:0>5}.tfit.pickle", "wb"
        #     ) as pickle_filepath:
        #         pickle.dump(tfit, pickle_filepath)
        #     # except:
        #     #     pass

        #     # try:
        #     new_hdul = mb.oned_spectrum_to_hdu(tfit=tfit)

        #     # print(tfit["coeffs"])
        #     # print(mb.N, len(tfit["coeffs"]))
        #     # exit()

        #     for k, v in mb.PA.items():
        #         for pa, beam_idx in v.items():
        #             # try:
        #             _mb = multifit.MultiBeam(
        #                 [mb.beams[i] for i in beam_idx], **beam_kwargs
        #             )
        #             _tfit = tfit.copy()
        #             _tfit["coeffs"] = [tfit["coeffs"][i] for i in beam_idx]
        #             _tfit["coeffs"].extend(tfit["coeffs"][mb.N :])
        #             _tfit["coeffs"] = np.asarray(_tfit["coeffs"])
        #             out = _mb.oned_spectrum_to_hdu(tfit=_tfit)
        #             out[-1].header["EXTVER"] = pa
        #             out[-1].header["FILTER"] = _mb.beams[0].grism.filter
        #             new_hdul.append(out[-1])
        #             # except:
        #             #     continue
        #             # print (out[0].header)
        #         # mb = MultiBeam()
        #         # print (v)
        #     # mb = MultiBeam()
        #     # try:
        #     #     del _mb
        #     # except:
        #     #     pass

        #     # new_hdul.info()

        #     new_hdul.writeto(Path.cwd() / f"{field_name}_{mb.id:0>5}.1D_RC.fits")
        #     del new_hdul
        #     del mb
        #     # except:
        #     #     pass

        #     print(f"{mpi_rank=}: Fit complete, output saved.")
        #     print(f"{mpi_rank=}: Time taken: {time()-t0}")
        #     for filetype in filetype_list:
        #         [
        #             p.rename(extractions_dir / filetype / p.name)
        #             for p in Path.cwd().glob(f"*{obj_id}.*{filetype}*")
        #         ]
        #     # (Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits").rename(
        #     #     extractions_dir
        #     #     / "beams_stacked"
        #     #     / f"{field_name}_{obj_id:0>5}.beams.fits"
        #     # )
        # except Exception as e:
        #     print(f"{mpi_rank=}: Fitting failed for {obj_id}: {e}")
