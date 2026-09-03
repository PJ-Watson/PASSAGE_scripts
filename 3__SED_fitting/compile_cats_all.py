"""A tool to compile the best available SED fits."""

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table, join, vstack

parser = argparse.ArgumentParser(
    description="Compile multiple versions of the SED fitting catalogues."
)
parser.add_argument(
    "cat_dir",
    type=str,
    metavar="cat_dir",
    help="The path of the directory in which the output catalogues are stored.",
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
    # action="extend",
    nargs="+",
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
parser.add_argument(
    "--overwrite",
    action=argparse.BooleanOptionalAction,
    help="Overwrite an existing catalogue.",
)

if __name__ == "__main__":

    args = parser.parse_args()

    cat_dir = Path(args.cat_dir)

    out_path = cat_dir / args.out_name
    if (not (out_path.is_file())) or args.overwrite:

        sed_fits_tab = None

        min_bands = np.atleast_1d(args.min_bands)
        if len(min_bands) < len(args.pref_order):
            min_bands = np.repeat(min_bands, len(args.pref_order))

        for i, cat_name in enumerate(args.pref_order):
            tab = Table.read(cat_dir / f"SED_fits_{cat_name}.fits")
            tab = tab[tab["n_bands"] >= min_bands[i]]
            tab["source_cat"] = cat_name
            tab["field_id"] = [
                f"{f}-{i:0>5}" for f, i, in zip(tab["field"], tab["id_photcat"])
            ]
            if sed_fits_tab is None:
                sed_fits_tab = tab
            else:
                tab = tab[
                    np.logical_not(np.isin(tab["field_id"], sed_fits_tab["field_id"]))
                ]
                sed_fits_tab = vstack([sed_fits_tab, tab], join_type="outer")

        # Figure out later why this name is duplicated
        sed_fits_tab.remove_columns(["cosmoswebid", "cosmoswebid_1"])
        sed_fits_tab.rename_column("cosmoswebid_2", "cosmoswebid")

        full_cat = sed_fits_tab
        full_cat.sort(["field_id"])
        full_cat.meta["EXTNAME"] = "SED_FITS"

        full_cat["use_sed_flag"] = (
            (full_cat["n_bands"] >= 2) & (full_cat["n_lines"] >= 1)
        ) | (full_cat["n_bands"] >= 3)

        # Fuck it, easiest way to reorder it is to hardcode this
        # Probably won't be running these fields again anyway
        full_cat = full_cat[
            "field_id",
            "field",
            "id_photcat",
            "ra_photcat",
            "dec_photcat",
            "zbest",
            "zbesterr",
            "use_sed_flag",
            "id_huberty",
            "ra",
            "dec",
            "cosmos2020id",
            "cosmoswebid",
            "source_cat",
            "HST_WFC3_UVIS1.F475W_flux",
            "HST_WFC3_UVIS1.F475W_err",
            "HST_WFC3_UVIS1.F625W_flux",
            "HST_WFC3_UVIS1.F625W_err",
            "HST_ACS_WFC.F814W_flux",
            "HST_ACS_WFC.F814W_err",
            "jwst_nircam_f115w_flux",
            "jwst_nircam_f115w_err",
            "jwst_nircam_f150w_flux",
            "jwst_nircam_f150w_err",
            "jwst_nircam_f277w_flux",
            "jwst_nircam_f277w_err",
            "jwst_nircam_f444w_flux",
            "jwst_nircam_f444w_err",
            "jwst_miri_f770w_flux",
            "jwst_miri_f770w_err",
            "U.MP9302_flux",
            "U.MP9302_err",
            "hsc_g_v2018_flux",
            "hsc_g_v2018_err",
            "hsc_r2_v2018_flux",
            "hsc_r2_v2018_err",
            "hsc_i2_v2018_flux",
            "hsc_i2_v2018_err",
            "hsc_z_v2018_flux",
            "hsc_z_v2018_err",
            "hsc_y_v2018_flux",
            "hsc_y_v2018_err",
            "hsc_nb816_flux",
            "hsc_nb816_err",
            "hsc_nb921_flux",
            "hsc_nb921_err",
            "hsc_nb1010_flux",
            "hsc_nb1010_err",
            "Paranal_VISTA.Y_flux",
            "Paranal_VISTA.Y_err",
            "Paranal_VISTA.J_flux",
            "Paranal_VISTA.J_err",
            "Paranal_VISTA.H_flux",
            "Paranal_VISTA.H_err",
            "Paranal_VISTA.Ks_flux",
            "Paranal_VISTA.Ks_err",
            "Paranal_VISTA.NB118_flux",
            "Paranal_VISTA.NB118_err",
            "SupIA484_flux",
            "SupIA484_err",
            "SupIA527_flux",
            "SupIA527_err",
            "SupIA624_flux",
            "SupIA624_err",
            "SupIA679_flux",
            "SupIA679_err",
            "SupIA738_flux",
            "SupIA738_err",
            "SupIA767_flux",
            "SupIA767_err",
            "SupIA427_flux",
            "SupIA427_err",
            "SupIA505_flux",
            "SupIA505_err",
            "SupIA574_flux",
            "SupIA574_err",
            "SupIA709_flux",
            "SupIA709_err",
            "SupIA827_flux",
            "SupIA827_err",
            "NB711Suprime_flux",
            "NB711Suprime_err",
            "NB816Suprime_flux",
            "NB816Suprime_err",
            "cosmos2020farmerid",
            "U.MP9301_flux",
            "U.MP9301_err",
            "SupIB464_flux",
            "SupIB464_err",
            "Spitzer_IRAC.I1_flux",
            "Spitzer_IRAC.I1_err",
            "Spitzer_IRAC.I2_flux",
            "Spitzer_IRAC.I2_err",
            "Spitzer_IRAC.I3_flux",
            "Spitzer_IRAC.I3_err",
            "Spitzer_IRAC.I4_flux",
            "Spitzer_IRAC.I4_err",
            "GALEX_GALEX.NUV_flux",
            "GALEX_GALEX.NUV_err",
            "GALEX_GALEX.FUV_flux",
            "GALEX_GALEX.FUV_err",
            "jwst_niriss_f115w_flux",
            "jwst_niriss_f115w_err",
            "jwst_niriss_f150w_flux",
            "jwst_niriss_f150w_err",
            "jwst_niriss_f200w_flux",
            "jwst_niriss_f200w_err",
            "flux_auto",
            "flux_scale",
            "redshift",
            "redshift_error",
            "chisq",
            "fwhm",
            "fwhm_error",
            "flux_SII",
            "err_SII",
            "flux_Ha",
            "err_Ha",
            "flux_OIII",
            "err_OIII",
            "flux_Hb",
            "err_Hb",
            "flux_OII",
            "err_OII",
            "flux_NeIII-3867",
            "err_NeIII-3867",
            "contvz:dsfr1_16",
            "contvz:dsfr1_50",
            "contvz:dsfr1_84",
            "contvz:dsfr2_16",
            "contvz:dsfr2_50",
            "contvz:dsfr2_84",
            "contvz:dsfr3_16",
            "contvz:dsfr3_50",
            "contvz:dsfr3_84",
            "contvz:dsfr4_16",
            "contvz:dsfr4_50",
            "contvz:dsfr4_84",
            "contvz:dsfr5_16",
            "contvz:dsfr5_50",
            "contvz:dsfr5_84",
            "contvz:dsfr6_16",
            "contvz:dsfr6_50",
            "contvz:dsfr6_84",
            "contvz:massformed_16",
            "contvz:massformed_50",
            "contvz:massformed_84",
            "contvz:metallicity_16",
            "contvz:metallicity_50",
            "contvz:metallicity_84",
            "dust:Av_16",
            "dust:Av_50",
            "dust:Av_84",
            "nebular:logU_16",
            "nebular:logU_50",
            "nebular:logU_84",
            "redshift_16",
            "redshift_50",
            "redshift_84",
            "stellar_mass_16",
            "stellar_mass_50",
            "stellar_mass_84",
            "formed_mass_16",
            "formed_mass_50",
            "formed_mass_84",
            "sfr_16",
            "sfr_50",
            "sfr_84",
            "ssfr_16",
            "ssfr_50",
            "ssfr_84",
            "nsfr_16",
            "nsfr_50",
            "nsfr_84",
            "mass_weighted_age_16",
            "mass_weighted_age_50",
            "mass_weighted_age_84",
            "mass_weighted_zmet_16",
            "mass_weighted_zmet_50",
            "mass_weighted_zmet_84",
            "tform_16",
            "tform_50",
            "tform_84",
            "tquench_16",
            "tquench_50",
            "tquench_84",
            "UV_colour_16",
            "UV_colour_50",
            "UV_colour_84",
            "VJ_colour_16",
            "VJ_colour_50",
            "VJ_colour_84",
            "tform10_16",
            "tform10_50",
            "tform10_84",
            "tform50_16",
            "tform50_50",
            "tform50_84",
            "tform90_16",
            "tform90_50",
            "tform90_84",
            "sfr10_16",
            "sfr10_50",
            "sfr10_84",
            "sfr100_16",
            "sfr100_50",
            "sfr100_84",
            "input_redshift",
            "log_evidence",
            "log_evidence_err",
            "chisq_tot",
            "chisq_lines",
            "n_lines",
            "chisq_phot",
            "n_bands",
            "bin_edge_0",
            "bin_edge_1",
            "bin_edge_2",
            "bin_edge_3",
            "bin_edge_4",
            "bin_edge_5",
            "bin_edge_6",
            "bin_edge_7",
        ]

        full_cat.write(out_path, overwrite=True)

# python ./compile_cats_cosmos_copy.py
# /media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data/cats/
# passage_cosmos_redshift_catalog_v2.dat
# --pref_order v1.3.2_cosmosweb v1.3.2_cosmos2020 v1.3.2_cosmosweb v1.3.2_cosmos2020 v1.3.2
# --out_name=v1.3.2_best.fits
# --overwrite
# --min_bands 4 4 0 0 0
