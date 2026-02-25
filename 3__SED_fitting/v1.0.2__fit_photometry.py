"""Fit the PASSAGE/COSMOS fields (v1.0.2)."""

import os
import re
import zipfile
from functools import partial
from os import PathLike
from pathlib import Path

import astropy.units as u
import numpy as np
import sed_utils_old as sed_utils

# from bagpipes_fit_catalogue_varied_sfh import generate_fit_params
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, hstack, join, vstack
from numpy.typing import ArrayLike

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
# try:
#     from project_2025c import plotting_scripts

#     plotting_scripts.setup_aanda_style()
# except:
#     print("Could not setup A&A plot style.")

import warnings

# Of course this is a grizli problem
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter("ignore", category=VerifyWarning)
# def compile_passage_photcats(cat_dir : PathLike, par_ids = []):


if __name__ == "__main__":

    # Increment for newer versions of the catalogue
    fit_ver = "v1.0.2"

    fields = [
        "Par028",
        "Par003",
        "Par017",
        "Par023",
        "Par024",
        "Par025",
        "Par026",
        "Par029",
        "Par049",
        "Par051",
        "Par053",
        "Par052",
        "Par005",
        "Par006",
        "Par020",
    ]

    cat_versions = ["web", "2020"]

    out_base_dir = Path(os.getenv("ROOT_DIR")) / "2026_01_08__PASSAGE"
    passage_dir = out_base_dir / "PASSAGE_data"
    ref_cats_dir = out_base_dir / "ref_cats"
    filt_dir = out_base_dir / "transmission_curves"

    upload_dir = out_base_dir / "to_upload" / f"{fit_ver}"
    upload_dir.mkdir(exist_ok=True, parents=True)

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
                    out_base_dir,
                    passage_dir,
                    ref_cats_dir,
                    filt_dir,
                    fit_ver=fit_ver,
                    field=field,
                    cat_ver=cat_ver,
                    cosmos_id_name=cosmos_id_name,
                )

                bagpipes_input_cat = Table.read(
                    passage_dir
                    / field
                    / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}_extcorr.fits"
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

            init_pipes_params = {
                "redshift": (1.0, 1.0),
                "dust": {"type": "Calzetti", "Av": (0.0, 3.0), "eta": 2.0},
                "nebular": {"logU": (-3.5, -1.0)},
                "t_bc": 0.02,
            }

            basic_load_fn = partial(
                load_photom_bagpipes,
                phot_cat=bagpipes_input_cat_masked,
                zeropoint=23.9,
                id_colname="id_photcat",
                extra_frac_err=0.05,
                sci_suffix="_flux",
            )

            pipes_params = init_pipes_params.copy()
            pipes_params["contvz"] = {
                "massformed": (3.0, 12.0),
                "metallicity": (0.0, 3.0),
                "metallicity_prior_mu": 1.0,
                "metallicity_prior_sigma": 0.5,
                "bin_edges_low": [0.0, 30.0],
                "bin_edges_high": [-500.0, 0.0],
                "n_bins": 7,
                "dsfr1": (-10.0, 10.0),
                "dsfr1_prior": "student_t",
                "dsfr1_prior_scale": 0.5,
                "dsfr1_prior_df": 2,
                "dsfr2": (-10.0, 10.0),
                "dsfr2_prior": "student_t",
                "dsfr2_prior_scale": 0.5,
                "dsfr2_prior_df": 2,
                "dsfr3": (-10.0, 10.0),
                "dsfr3_prior": "student_t",
                "dsfr3_prior_scale": 0.5,
                "dsfr3_prior_df": 2,
                "dsfr4": (-10.0, 10.0),
                "dsfr4_prior": "student_t",
                "dsfr4_prior_scale": 0.5,
                "dsfr4_prior_df": 2,
                "dsfr5": (-10.0, 10.0),
                "dsfr5_prior": "student_t",
                "dsfr5_prior_scale": 0.5,
                "dsfr5_prior_df": 2,
                "dsfr6": (-10.0, 10.0),
                "dsfr6_prior": "student_t",
                "dsfr6_prior_scale": 0.5,
                "dsfr6_prior_df": 2,
            }

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
                # load_line_fluxes=(
                #     emlines_load_fn if run_type == "emlines" else None
                # ),
                mujy_plot=True,
                # lines_list=lines_list,
            )
            cat_fit.fit(n_live=400, verbose=True, mpi_serial=True, track_backlog=True)

            comm.Barrier()

            if rank == 0:

                if not (
                    passage_dir
                    / field
                    # / f"{field}_fit_{run_type}_{sfh_type}_cosmos{cat_ver}.fits"
                    / f"{field}_full_{fit_ver}_cosmos{cat_ver}.fits"
                ).is_file():

                    pipes_cat = Table.read(pipes_dir / "cats" / f"{run_name}.fits")
                    pipes_cat.rename_column("#ID", "id_photcat")
                    pipes_cat["id_photcat"] = pipes_cat["id_photcat"].astype(int)

                    pipes_cat.write(
                        passage_dir / field / f"{run_name}.fits", overwrite=True
                    )

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
                    passage_dir / "cats" / "passage_cosmos_redshift_catalog_v2.dat",
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
                # for file in passage_dir.glob(
                #     f"**/*_full_{fit_ver}_cosmos{cat_ver}.fits"
                # ):
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

                # print (tables)
                # exit()

                SED_fits = vstack(tables, join_type="outer")

                # SED_fits.pprint()
                # exit()

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

        # exit()
