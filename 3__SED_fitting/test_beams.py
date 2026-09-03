from pathlib import Path
import numpy as np
import tomllib
import os
from astropy.table import Table
from time import time

MPI_avail = False
# try:
#     from mpi4py import MPI

#     comm = MPI.COMM_WORLD
#     mpi_rank = comm.Get_rank()
#     mpi_size = comm.Get_size()

#     MPI_avail = True

# except ImportError:
#     print("Could not import MPI")
mpi_rank = 0
mpi_size = 1

if __name__ == "__main__":

    use_obj_id = 2074

    beams_path = Path(
        "/media/sharedData/data/2025_12_06_glass-a2744/grizli_home"
        f"/Extractions-NGDEEP_custom/beams/glass-a2744_{use_obj_id:0>5}.beams.fits"
    )

    config_path = Path(
        "/media/sharedData/python/py3.13_PIE/code"
        "/PASSAGE_scripts/1__extraction/config_files/CINECA_config_par682.toml"
    )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    import warnings

    # Quiet some of the common the grizli-induced warnings
    from astropy.io.fits.verify import VerifyWarning
    from astropy.units import UnitsWarning
    from astropy.wcs import FITSFixedWarning

    for w in [VerifyWarning, FITSFixedWarning, UnitsWarning, RuntimeWarning]:
        warnings.simplefilter("ignore", category=w)

    import grizli
    from astropy.io import fits
    from grizli import fitting, jwst_utils, multifit, prep, utils
    from grizli.pipeline import auto_script

    print("Grizli version: ", grizli.__version__)

    import shutil

    extractions_dir = (
        Path("/media/sharedData/data/2025_12_06_glass-a2744/grizli_home")
        / "test_photom_scaling"
    )
    extractions_dir.mkdir(exist_ok=True)

    shutil.copy2(beams_path, extractions_dir / "beams" / beams_path.name)

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

    kwargs = auto_script.get_yml_parameters()

    beam_kwargs = config["extraction"].get("beams", {})

    field_name = "glass-a2744"

    if mpi_rank == 0:
        os.chdir(extractions_dir)
        args = auto_script.generate_fit_params(
            field_root=field_name,
            **beam_kwargs,
            **config["extraction"].get("fit_params", {}),
            # **dict(
            #     scale_photometry=True,
            #     # phot
            # )
        )
    else:
        args = None

    if MPI_avail:
        args = comm.bcast(args, root=0)
        comm.Barrier()

    # for i, row in enumerate(fit_cat[:]):
    from eazy import filters, utils

    res = filters.FilterFile(
        "/media/sharedData/python/py3.13_PIE/code/eazy-py/eazy/data/eazy-photoz/filters/FILTER.RES.latest"
    )

    phot_cat = Table.read(extractions_dir / f"{field_name}_phot.fits")

    phot_master = dict(filters=[], flam=[], eflam=[])

    for filt in ["f115w", "f150w", "f200w"]:

        filt_def = res.filters[res.search(f"niriss-{filt}")[-1]]
        phot_cat[f"{filt}n_flux_auto_flam"] = phot_cat[f"{filt}n_flux_auto"] * (
            10**-29 * 2.9979 * 10**18 / filt_def.pivot**2
        )
        phot_cat[f"{filt}n_flux_auto_eflam"] = phot_cat[f"{filt}n_fluxerr_auto"] * (
            10**-29 * 2.9979 * 10**18 / filt_def.pivot**2
        )

        phot_master["filters"].append(filt_def)

    # for obj_id in [3070]:
    for row in phot_cat:

        t0 = time()

        obj_id = row["id"]

        # print(f"{mpi_rank=}, {obj_id=}")

        if obj_id not in [use_obj_id]:
            continue

        phot = phot_master.copy()

        for filt in ["f115w", "f150w", "f200w"]:

            phot["flam"].append(row[f"{filt}n_flux_auto_flam"])
            phot["eflam"].append(row[f"{filt}n_flux_auto_eflam"])

        # print (phot)
        # exit()
        # # try:
        phot["flam"] = np.asarray(phot["flam"])
        phot["eflam"] = np.asarray(phot["eflam"])
        phot["filters"] = np.asarray(phot["filters"])

        os.chdir(extractions_dir)

        try:
            mb = multifit.MultiBeam(
                str(
                    extractions_dir / "beams" / f"{field_name}_{obj_id:0>5}.beams.fits"
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

        # if config["extraction"].get("stack_beams", False):
        #     try:
        #         shutil.copy2(
        #             extractions_dir
        #             / "beams_stacked"
        #             / f"{field_name}_{obj_id:0>5}.beams.fits",
        #             f"{field_name}_{obj_id:05}.beams.fits",
        #         )
        #     except:
        #         # Cluster and stack the individual beams before fitting
        #         new_mb = gen_stacked_beams(
        #             mb,
        #             group_name=field_name,
        #             **config["extraction"].get("stack_kwargs", {}),
        #             **beam_kwargs,
        #         )

        #         # Save and copy immediately to the stacked folder
        #         # Avoids rerunning clustering code if things crash
        #         # during redshift fitting
        #         new_mb.write_master_fits()
        #         shutil.copy2(
        #             Path.cwd() / f"{field_name}_{obj_id:0>5}.beams.fits",
        #             extractions_dir
        #             / "beams_stacked"
        #             / f"{field_name}_{obj_id:0>5}.beams.fits",
        #         )
        #         del new_mb

        del mb

        # Change parameters here for the drizzled emission line outputs
        pline = args.get("pline", {})

        if config["extraction"].get("recalculate_size", True):
            # pline["size"] = int(np.clip(2 * row["beam_size"] * 0.06, a_min=3, a_max=30))
            pline["size"] = int(np.clip(2 * 0 * 0.06, a_min=3, a_max=30))
            # pline["size"] = int(np.clip(2 *  * 0.06, a_min=3, a_max=30))
            pline["size"] = 8

        mb, st, fit, tfit, line_hdu = fitting.run_all_parallel(
            int(obj_id),
            pline=pline,
            get_output_data=True,
            phot=phot,
            scale_photometry=2,
            protect=False,
            zr=[1.3, 1.4],
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
