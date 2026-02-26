"""The primary script for performing SED fitting in the PASSAGE/COSMOS fields."""

import os
import re
import zipfile
from functools import partial
from os import PathLike
from pathlib import Path

import numpy as np
import sed_utils
from astropy.table import Table, hstack, join, vstack

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    print("Could not import MPI")
    rank = 0
    size = 1

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
    }
)

import warnings

# Silence common warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astropy.wcs import FITSFixedWarning

for w in [VerifyWarning, FITSFixedWarning, UnitsWarning]:
    warnings.simplefilter("ignore", category=w)

import argparse
import tomllib

parser = argparse.ArgumentParser(
    description="Perform an SED fit to all galaxies in the PASSAGE/COSMOS fields."
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

    fit_ver = config["general"]["fit_ver"]
    cat_versions = np.atleast_1d(config["general"].get("cat_versions", ["web"]))
    fields = np.atleast_1d(config["general"].get("fields", ["Par028"]))

    out_base_dir = Path(config["files"].get("out_base_dir", Path.cwd()))
    passage_dir = out_base_dir / config["files"].get("passage_dir", "PASSAGE_data")
    ref_cats_dir = out_base_dir / config["files"].get("ref_cats_dir", "ref_cats")
    filt_dir = out_base_dir / config["files"].get("filt_dir", "transmission_curves")

    upload_dir = (
        out_base_dir / config["files"].get("upload_dir", "to_upload") / f"{fit_ver}"
    )
    upload_dir.mkdir(exist_ok=True, parents=True)

    # Since TOML cannot include tuples, but bagpipes expects priors to be
    # tuple type, this is a roundabout way of finding a solution.
    # Obviously, check the config file before blindly calling `eval()` on
    # something you didn't write yourself.
    # pipes_params = eval(
    #     str(config["fit_instructions"]).replace("[", "(").replace("]", ")")
    # )

    pipes_params = sed_utils.correct_pipes_params(config["fit_instructions"])

    for cat_ver in cat_versions:

        if cat_ver == "2020":
            cosmos_id_name = "cosmos2020farmerid"
        else:
            cosmos_id_name = "cosmoswebid"

        summary_plot_archive = (
            upload_dir / f"SED_fits_summary_plots_{fit_ver}_cosmos{cat_ver}.zip"
        )

        for field in fields:

            if rank == 0:
                sed_utils.prepare_catalogues(
                    config,
                    passage_dir,
                    ref_cats_dir,
                    filt_dir,
                    fit_ver=fit_ver,
                    field=field,
                    cat_ver=cat_ver,
                    cosmos_id_name=cosmos_id_name,
                )

            pipes_dir = passage_dir / "pipes"
            pipes_dir.mkdir(exist_ok=True, parents=True)
            os.chdir(pipes_dir.parent)

            comm.Barrier()

            bagpipes_input_cat = Table.read(
                passage_dir
                / field
                / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}_extcorr.fits"
            )
            filter_list = np.loadtxt(
                passage_dir
                / field
                / f"{field}_filter_list_{fit_ver}_cosmos{cat_ver}.txt",
                dtype=str,
            )

            bagpipes_input_cat_masked = bagpipes_input_cat[
                bagpipes_input_cat[cosmos_id_name] > 0
            ]

            if len(bagpipes_input_cat_masked) == 0:
                if rank == 0:
                    print(f"No objects to fit in field {field} for {cat_ver=}.")
                continue

            try:
                from bagpipes_extended.pipeline import (
                    load_lines_bagpipes,
                    load_photom_bagpipes,
                )
            except:
                raise ImportError(
                    "Please install the `bagpipes-extended` library from "
                    "https://github.com/PJ-Watson/bagpipes-extended"
                )

            basic_load_fn = partial(
                load_photom_bagpipes,
                phot_cat=bagpipes_input_cat_masked,
                zeropoint=23.9,
                id_colname="id_photcat",
                extra_frac_err=0.05,
                sci_suffix="_flux",
            )

            emlines_load_fn = None
            if config["general"].get("fit_emlines", False):
                bagpipes_input_emlines_cat = Table.read(
                    passage_dir
                    / field
                    / f"{field}_bagpipes_input_emlines_{fit_ver}_cosmos{cat_ver}.fits"
                )
                bagpipes_input_emlines_cat_masked = bagpipes_input_emlines_cat[
                    bagpipes_input_cat[cosmos_id_name] > 0
                ]

                emlines_load_fn = partial(
                    load_lines_bagpipes,
                    line_mapping={
                        k: v
                        for k, v in sed_utils.GRIZLI_TO_CLOUDY_NAMES_ONLY.items()
                        if k
                        in config["general"].get(
                            "line_names", sed_utils.DEFAULT_FIT_LINES
                        )
                    },
                    line_cat=bagpipes_input_emlines_cat_masked,
                    id_colname="id_photcat",
                )

            run_name = f"{field}_fit_{fit_ver}_cosmos{cat_ver}"

            IDs_to_use = bagpipes_input_cat_masked["id_photcat"]
            IDs_to_use = IDs_to_use[bagpipes_input_cat_masked[cosmos_id_name] > 0]

            import bagpipes
            from bagpipes_extended.sed.continuity_varied_z import (
                contvz,
            )
            from bagpipes_extended.sed.fit_catalogue_varied_sfh import (
                fit_catalogue,
            )

            bagpipes.models.star_formation_history.contvz = contvz

            cat_fit = fit_catalogue(
                IDs=IDs_to_use,
                fit_instructions=pipes_params,
                load_data=basic_load_fn,
                spectrum_exists=False,
                make_plots=True,
                cat_filt_list=filter_list,
                redshifts=bagpipes_input_cat_masked["zbest"],
                redshift_sigma=np.max(
                    [
                        bagpipes_input_cat_masked["zbesterr"],
                        np.full_like(bagpipes_input_cat_masked["zbesterr"], 0.001),
                    ]
                ),
                run=run_name,
                full_catalogue=True,
                load_line_fluxes=emlines_load_fn,
                mujy_plot=True,
            )
            cat_fit.fit(n_live=400, verbose=True, mpi_serial=True, track_backlog=True)

            comm.Barrier()

            if rank == 0:

                if not (
                    passage_dir / field / f"{field}_full_{fit_ver}_cosmos{cat_ver}.fits"
                ).is_file():

                    pipes_cat = Table.read(pipes_dir / "cats" / f"{run_name}.fits")
                    pipes_cat.rename_column("#ID", "id_photcat")
                    pipes_cat["id_photcat"] = pipes_cat["id_photcat"].astype(int)

                    pipes_cat.write(
                        passage_dir / field / f"{run_name}.fits", overwrite=True
                    )

                    try:
                        bagpipes_input_cat = join(
                            bagpipes_input_cat,
                            bagpipes_input_emlines_cat,
                            keys="id_photcat",
                            join_type="left",
                            table_names=["phot", "emlines"],
                        )
                    except Exception as e:
                        print(e)
                        pass

                    total_cat = join(bagpipes_input_cat, pipes_cat, join_type="left")

                    for k, v in list(total_cat.meta.items()):
                        total_cat.meta.pop(k)
                    total_cat.meta["EXTNAME"] = "SED_FITTING"
                    total_cat.sort("id_photcat")
                    total_cat.write(
                        passage_dir
                        / field
                        / f"{field}_full_{fit_ver}_cosmos{cat_ver}.fits",
                        overwrite=True,
                    )

                with zipfile.ZipFile(summary_plot_archive, "a") as myzip:
                    zip_path = zipfile.Path(myzip)
                    for f in (pipes_dir / "plots" / run_name).glob("*summary.pdf"):
                        if not (zip_path / f.relative_to(pipes_dir / "plots")).exists():
                            myzip.write(f, f.relative_to(pipes_dir / "plots"))

        if rank == 0:

            cat_path_1 = (
                passage_dir / "cats" / f"SED_fits_{fit_ver}_cosmos{cat_ver}.fits"
            )
            cat_path_2 = upload_dir / f"SED_fits_{fit_ver}_cosmos{cat_ver}.fits"
            if not (cat_path_1.is_file() and cat_path_2.is_file()):

                linefinding_cat = Table.read(
                    passage_dir
                    / "cats"
                    / config["catalogues"].get(
                        "passage_cat_name", "passage_cosmos_redshift_catalog_v2.dat"
                    ),
                    format="ascii.tab",
                )
                passage_z_cat = linefinding_cat[
                    "id",
                    "ra",
                    "dec",
                    "field",
                    "zbest",
                    "zbesterr",
                    "cosmoswebid",
                    "cosmos2020id",
                ]
                passage_z_cat.rename_column("id", "id_huberty")

                tables = []
                for field in fields:
                    file = (
                        passage_dir
                        / f"{field}"
                        / f"{field}_full_{fit_ver}_cosmos{cat_ver}.fits"
                    )
                    try:
                        _tab = Table.read(file)
                    except:
                        _tab = Table.read(
                            file.parent
                            / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}_extcorr.fits"
                        )
                    _tab["field"] = file.name.split("_")[0]
                    tables.append(_tab)

                SED_fits = vstack(tables, join_type="outer")

                full_cat = join(
                    passage_z_cat,
                    SED_fits,
                    keys=["field", "id_huberty", "zbest", "zbesterr"],
                    join_type="left",
                )

                full_cat.sort("id_huberty")
                full_cat.meta["EXTNAME"] = f"SED_FITTING_{fit_ver}"
                full_cat.write(cat_path_1, overwrite=True)
                full_cat.write(cat_path_2, overwrite=True)

                print(f"Finished for {cat_ver=}.")

            full_dir_archive = (
                upload_dir / f"full_dir_archive_{fit_ver}_cosmos{cat_ver}.zip"
            )

            with zipfile.ZipFile(full_dir_archive, "a") as myzip:
                zip_path = zipfile.Path(myzip)
                for f in passage_dir.glob(f"**/*{fit_ver}_cosmos{cat_ver}*"):
                    if f.is_dir():
                        for subfiles in f.glob("*"):
                            if not (
                                zip_path / subfiles.relative_to(out_base_dir)
                            ).exists():
                                myzip.write(
                                    subfiles, subfiles.relative_to(out_base_dir)
                                )
                    if not (zip_path / f.relative_to(out_base_dir)).exists():
                        myzip.write(f, f.relative_to(out_base_dir))
                for f in filt_dir.glob("*"):
                    if not (zip_path / f.relative_to(out_base_dir)).exists():
                        myzip.write(f, f.relative_to(out_base_dir))
            # for file_path in passage_dir.glob(f"**/*{fit_ver}_cosmos{cat_ver}*"):
            #     print (file_path)
            #     print (file_path.is_dir())
            # exit()
