"""A tool to compile the best available SED fits."""

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table, join, vstack

parser = argparse.ArgumentParser(
    description="Make comparison plots of SED fit parameters."
)
parser.add_argument(
    "cat_dir",
    type=str,
    metavar="cat_dir",
    help="The path of the directory in which the output catalogues are stored.",
)
parser.add_argument(
    "ref_cat",
    type=str,
    metavar="ref_cat",
    help="The reference catalogue containing all galaxies.",
)
parser.add_argument(
    "--pref_order",
    action="extend",
    nargs="+",
    type=str,
    help="The order of preference for adding results from catalogues.",
)
parser.add_argument(
    "--min_bands",
    type=int,
    default=2,
    help="The minimum number of bands for the fit to be included.",
)
parser.add_argument(
    "--out_name",
    type=str,
    default="SED_fits_v1.0.3_best.fits",
    help="The output catalogue name.",
)

if __name__ == "__main__":

    args = parser.parse_args()

    cat_dir = Path(args.cat_dir)

    out_path = cat_dir / args.out_name
    if not (out_path.is_file()):

        linefinding_cat = Table.read(
            cat_dir / args.ref_cat,
            format="ascii.tab",
        )
        passage_z_cat = linefinding_cat[
            "id",
            # "ra",
            # "dec",
            "field",
            "zbest",
            "zbesterr",
            # "cosmoswebid",
            # "cosmos2020id",
        ]
        passage_z_cat.rename_column("id", "id_huberty")

        sed_fits_tab = None

        for cat_name in args.pref_order:
            # print (cat_name)
            tab = Table.read(cat_dir / f"SED_fits_{cat_name}.fits")
            tab = tab[tab["n_bands"] >= args.min_bands]
            tab["source_cat"] = cat_name
            if sed_fits_tab is None:
                sed_fits_tab = tab
            else:
                tab = tab[
                    np.logical_not(
                        np.isin(tab["id_huberty"], sed_fits_tab["id_huberty"])
                    )
                ]
                # tab.pprint()
                sed_fits_tab = vstack([sed_fits_tab, tab], join_type="outer")

        # sed_fits_tab = sed_fits_tab[sed_fits_tab["cosmoswebid_1"]>0]
        # sed_fits_tab["cosmoswebid_2","cosmoswebid_1"].pprint()

        # Figure out later why this name is duplicated
        sed_fits_tab.remove_columns(["cosmoswebid", "cosmoswebid_1"])
        sed_fits_tab.rename_column("cosmoswebid_2", "cosmoswebid")

        sed_fits_tab.pprint()

        full_cat = join(
            passage_z_cat,
            sed_fits_tab,
            keys=["field", "id_huberty", "zbest", "zbesterr"],
            join_type="left",
        )

        full_cat.sort("id_huberty")
        full_cat.pprint()
        full_cat.meta["EXTNAME"] = "SED_FITS"
        # full_cat.write(cat_path_1, overwrite=True)
        full_cat.write(out_path, overwrite=True)

        # print(f"Finished for {cat_ver=}.")
