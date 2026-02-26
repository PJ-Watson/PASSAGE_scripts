"""A collection of utility functions and constants to aid in SED fitting."""

cosmosweb_name_mapping = {
    # NIRISS
    "f115wn": "jwst_niriss_f115w",
    "f150wn": "jwst_niriss_f150w",
    "f200wn": "jwst_niriss_f200w",
    # COSMOS-Web
    "f115w": "jwst_nircam_f115w",
    "f150w": "jwst_nircam_f150w",
    "f277w": "jwst_nircam_f277w",
    "f444w": "jwst_nircam_f444w",
    "hst-f814w": "HST_ACS_WFC.F814W",
    "f770w": "jwst_miri_f770w",
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html
    # The same as on SVO
    "cfht-u": "U.MP9302",
    # https://hsc.mtk.nao.ac.jp/pipedoc/pipedoc_8_e/hsc_info_e/index.html
    "hsc-g": "hsc_g_v2018",
    "hsc-r": "hsc_r2_v2018",
    "hsc-i": "hsc_i2_v2018",
    "hsc-z": "hsc_z_v2018",
    "hsc-y": "hsc_y_v2018",
    "hsc-nb0816": "hsc_nb816",
    "hsc-nb0921": "hsc_nb921",
    "hsc-nb1010": "hsc_nb1010",
    # SVO
    "uvista-y": "Paranal_VISTA.Y",
    "uvista-j": "Paranal_VISTA.J",
    "uvista-h": "Paranal_VISTA.H",
    "uvista-ks": "Paranal_VISTA.Ks",
    "uvista-nb118": "Paranal_VISTA.NB118",
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/scla/docs/filt.html
    "sc-ia484": "SupIA484",
    "sc-ia527": "SupIA527",
    "sc-ia624": "SupIA624",
    "sc-ia679": "SupIA679",
    "sc-ia738": "SupIA738",
    "sc-ia767": "SupIA767",
    "sc-ib427": "SupIA427",
    "sc-ib505": "SupIA505",
    "sc-ib574": "SupIA574",
    "sc-ib709": "SupIA709",
    "sc-ib827": "SupIA827",
    "sc-nb711": "NB711Suprime",
    "sc-nb816": "NB816Suprime",
    # SVO
    "irac-ch1": "Spitzer_IRAC.I1",
    "irac-ch2": "Spitzer_IRAC.I2",
    "irac-ch3": "Spitzer_IRAC.I3",
    "irac-ch4": "Spitzer_IRAC.I4",
}

cosmos2020_name_mapping = {
    # NIRISS
    "f115wn": "jwst_niriss_f115w",
    "f150wn": "jwst_niriss_f150w",
    "f200wn": "jwst_niriss_f200w",
    # SVO
    "acs_f814w": "HST_ACS_WFC.F814W",
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html
    # The same as on SVO
    "cfht_u": "U.MP9302",
    "cfht_ustar": "U.MP9301",
    # https://hsc.mtk.nao.ac.jp/pipedoc/pipedoc_8_e/hsc_info_e/index.html
    "hsc_g": "hsc_g_v2018",
    "hsc_r": "hsc_r2_v2018",
    "hsc_i": "hsc_i2_v2018",
    "hsc_z": "hsc_z_v2018",
    "hsc_y": "hsc_y_v2018",
    "hsc-nb0816": "hsc_nb816",
    "hsc-nb0921": "hsc_nb921",
    "hsc-nb1010": "hsc_nb1010",
    # SVO
    "uvista_y": "Paranal_VISTA.Y",
    "uvista_j": "Paranal_VISTA.J",
    "uvista_h": "Paranal_VISTA.H",
    "uvista_ks": "Paranal_VISTA.Ks",
    "uvista_nb118": "Paranal_VISTA.NB118",
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/scla/docs/filt.html
    "sc_ia484": "SupIA484",
    "sc_ia527": "SupIA527",
    "sc_ia624": "SupIA624",
    "sc_ia679": "SupIA679",
    "sc_ia738": "SupIA738",
    "sc_ia767": "SupIA767",
    "sc_ib427": "SupIA427",
    "sc_ib464": "SupIB464",
    "sc_ib505": "SupIA505",
    "sc_ib574": "SupIA574",
    "sc_ib709": "SupIA709",
    "sc_ib827": "SupIA827",
    "sc_nb711": "NB711Suprime",
    "sc_nb816": "NB816Suprime",
    # SVO
    "splash_ch1": "Spitzer_IRAC.I1",
    "splash_ch2": "Spitzer_IRAC.I2",
    "splash_ch3": "Spitzer_IRAC.I3",
    "splash_ch4": "Spitzer_IRAC.I4",
    # SVO
    "irac_ch1": "Spitzer_IRAC.I1",
    "irac_ch2": "Spitzer_IRAC.I2",
    "irac_ch3": "Spitzer_IRAC.I3",
    "irac_ch4": "Spitzer_IRAC.I4",
    # SVO
    "galex_nuv": "GALEX_GALEX.NUV",
    "galex_fuv": "GALEX_GALEX.FUV",
}

inv_cosmosweb = {v: k for k, v in cosmosweb_name_mapping.items()}
inv_cosmos2020 = {v: k for k, v in cosmos2020_name_mapping.items()}

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, join, vstack
from numpy.typing import ArrayLike


def apply_dust_correction(
    phot_cat: Table,
    filter_list: ArrayLike,
    ra_colname: str = "ra_photcat",
    dec_colname: str = "dec_photcat",
) -> Table:
    """
    Corrects a photometric catalogue for galactic extinction.

    Parameters
    ----------
    phot_cat : Table
        The input photometric catalogue.
    filter_list : ArrayLike
        The list of filter transmission curves used in the catalogue, in
        the same format as for `bagpipes`.
    ra_colname : str, optional
        The column name for the RA coordinate, by default `"ra_photcat"`.
    dec_colname : str, optional
        The column name for the Dec. coordinate, by default
        `"dec_photcat"`.

    Returns
    -------
    Table
        The extinction-corrected catalogue.
    """

    from astropy.coordinates import SkyCoord
    from bagpipes.filters import filter_set

    # from dustmaps.edenhofer2023 import Edenhofer2023Query
    # dust_map = Edenhofer2023Query(integrated=True)
    from dustmaps.sfd import SFDQuery

    dust_map = SFDQuery()

    from dust_extinction.parameter_averages import G23

    ext = G23(Rv=3.1)

    coords = SkyCoord(
        ra=phot_cat[ra_colname],
        dec=phot_cat[dec_colname],
        unit="deg",
    )
    ebv = dust_map(coords)

    filts = filter_set(filter_list)

    for f_name, f_lam in zip(filter_list, filts.eff_wavs):
        print(
            Path(f_name).stem,
            f_lam / 1e4,
            ext.extinguish(f_lam * u.Angstrom, Ebv=ebv[0]),
        )
        phot_cat[f"{Path(f_name).stem}_flux"] /= ext.extinguish(
            f_lam * u.Angstrom, Ebv=ebv
        )
        phot_cat[f"{Path(f_name).stem}_err"] /= ext.extinguish(
            f_lam * u.Angstrom, Ebv=ebv
        )

    return phot_cat


def prepare_catalogues(
    config: dict,
    passage_dir: Path,
    ref_cats_dir: Path,
    filt_dir: Path,
    fit_ver: str = "v1.1.0",
    field: str = "Par028",
    cat_ver: str = "web",
    cosmos_id_name: str = "cosmoswebid",
) -> None:
    """
    Prepare the catalogues for SED fitting.

    Parameters
    ----------
    config : dict
        The dictionary storing the configuration used for this iteration
        of SED fitting.
    passage_dir : Path
        The directory containing all existing phot/spec cats in
        field-specific subdirectories.
    ref_cats_dir : Path
        The directory containing the reference COSMOS2020/Web catalogues.
    filt_dir : Path
        The directory storing the transmission curves for the filters in
        the photometric catalogues.
    fit_ver : str, optional
        The string identifying the semantic version of the fit, by default
        `"v1.1.0"`.
    field : str, optional
        The string identifying the PASSAGE field to fit, by default
        `"Par028"`.
    cat_ver : str, optional
        The version of the reference catalogue to use, by default `"web"`.
    cosmos_id_name : str, optional
        The column ID matching the ID in the reference catalogue, by
        default `"cosmoswebid"`.
    """

    # Check directories exist (but if not, things are likely to fail anyway)
    passage_dir.mkdir(exist_ok=True, parents=True)
    ref_cats_dir.mkdir(exist_ok=True, parents=True)
    filt_dir.mkdir(exist_ok=True, parents=True)

    max_sep = config["general"].get("max_sep", 0.3) * u.arcsec

    try:
        passage_matched_phot = Table.read(
            passage_dir / field / f"{field}_matched_phot_{fit_ver}_cosmos{cat_ver}.fits"
        )
    except:

        try:
            passage_matched = Table.read(
                passage_dir
                / "cats"
                / f"passage_matched_phot_{fit_ver}_cosmos{cat_ver}.fits"
            )
        except:
            passage_z_cat = Table.read(
                passage_dir
                / "cats"
                / config["catalogues"].get(
                    "passage_cat_name", "passage_cosmos_redshift_catalog_v2.dat"
                ),
                format="ascii.tab",
            )[
                "id",
                "ra",
                "dec",
                "field",
                "field_id",
                "zbest",
                "zbesterr",
                "cosmoswebid",
            ]

            if cat_ver == "2020":

                cosmos2020_cat = Table.read(
                    ref_cats_dir
                    / config["catalogues"].get(
                        "2020_cat_name", "COSMOS2020_FARMER_R1_v2.2_p3.fits"
                    ),
                    hdu=config["catalogues"].get("2020_phot_hdu", "PHASE3CATALOG"),
                )

                passage_coords = SkyCoord(
                    ra=passage_z_cat["ra"],
                    dec=passage_z_cat["dec"],
                    unit="deg",
                )
                cosmos2020_coords = SkyCoord(
                    ra=cosmos2020_cat["ALPHA_J2000"],
                    dec=cosmos2020_cat["DELTA_J2000"],
                )

                idx, d2d, d3d = passage_coords.match_to_catalog_sky(cosmos2020_coords)
                sep_constraint = d2d < max_sep
                passage_matches = passage_coords[sep_constraint]
                cosmos2020_matches = cosmos2020_cat[idx[sep_constraint]]
                cosmos2020_matches.rename_column("ID", "cosmos2020farmerid")
                cosmos2020_matches["passageid"] = passage_z_cat["id"][sep_constraint]
                cosmos2020_matches.rename_columns(
                    cosmos2020_matches.colnames,
                    [c.lower() for c in cosmos2020_matches.colnames],
                )

                passage_matched = join(
                    passage_z_cat,
                    cosmos2020_matches,
                    keys_left="id",
                    keys_right="passageid",
                    join_type="left",
                    keep_order=True,
                )

            else:
                cosmos_cat = Table.read(
                    ref_cats_dir
                    / config["catalogues"].get(
                        "web_cat_name", "COSMOSWeb_mastercatalog_v1.1.fits"
                    ),
                    hdu=config["catalogues"].get(
                        "web_phot_hdu", "PHOTOMETRY HOTCOLD AND SE++"
                    ),
                )
                passage_z_cat["cosmoswebid"] = passage_z_cat["cosmoswebid"].astype(int)

                cosmos_cat.rename_columns(
                    ["id", "ra", "dec"],
                    ["cosmoswebid", "ra_cosmosweb", "dec_cosmosweb"],
                )

                passage_matched = join(
                    passage_z_cat,
                    cosmos_cat,
                    keys="cosmoswebid",
                    keep_order=True,
                    join_type="left",
                )

            passage_matched.write(
                passage_dir
                / "cats"
                / f"passage_matched_phot_{fit_ver}_cosmos{cat_ver}.fits"
            )

        passage_matched = passage_matched[passage_matched["field"] == field]

        field_phot = Table.read(passage_dir / field / f"{field}_photcat.fits")

        field_phot["id_photcat"] = field_phot["id"].astype(int)
        field_phot.remove_column("id")
        passage_matched["id_photcat"] = passage_matched["field_id"].astype(int)
        passage_matched.rename_column("id", "id_huberty")

        passage_matched_phot = join(
            passage_matched,
            field_phot,
            keys="id_photcat",
            table_names=["huberty", "photcat"],
        )

        passage_matched_phot.write(
            passage_dir / field / f"{field}_matched_phot_{fit_ver}_cosmos{cat_ver}.fits"
        )

    try:
        phot_cat = Table.read(
            passage_dir
            / field
            / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}.fits"
        )
        filter_list = np.loadtxt(
            passage_dir / field / f"{field}_filter_list_{fit_ver}_cosmos{cat_ver}.txt",
            dtype=str,
        )
    except:
        passage_matched_phot["id_photcat"] = passage_matched_phot["id_photcat"].astype(
            int
        )

        phot_cat = passage_matched_phot[
            "id_photcat",
            "id_huberty",
            "zbest",
            "zbesterr",
            "ra_photcat",
            "dec_photcat",
            "flux_auto",
        ]
        phot_cat[cosmos_id_name] = passage_matched_phot[cosmos_id_name]
        filter_list = []

        for c in passage_matched_phot.colnames:
            # Drop IRAC bands in Cosmos-web fits
            if ("irac" in c.lower()) and (cat_ver == "web"):
                continue
            if c.endswith("_flux_auto"):
                cat_filt = c.removesuffix("_flux_auto") + "n"
                cat_filt = cosmosweb_name_mapping[cat_filt]
            elif c.startswith("flux_model_"):
                cat_filt = c.removeprefix("flux_model_")
                cat_filt = cosmosweb_name_mapping[cat_filt]
            elif (
                c.endswith("_flux")
                and not (c.endswith("wn_flux"))
                and ("splash" not in c)
            ):
                cat_filt = c.removesuffix("_flux")
                cat_filt = cosmos2020_name_mapping[cat_filt]
            else:
                continue

            filter_list.append(str(filt_dir / f"{cat_filt}.dat"))

            # print (c, cat_filt)

            phot_cat[f"{cat_filt}_flux"] = passage_matched_phot[c]
            try:
                phot_cat[f"{cat_filt}_err"] = passage_matched_phot[
                    f"flux_err-cal_model_{c.removeprefix("flux_model_")}"
                ]
            except:
                try:
                    phot_cat[f"{cat_filt}_err"] = passage_matched_phot[
                        f"{c.removesuffix("_flux")}_fluxerr"
                    ]
                except:
                    phot_cat[f"{cat_filt}_err"] = passage_matched_phot[
                        f"{c.removesuffix("_flux_auto")}_fluxerr_auto"
                    ]

        uniq, uniq_ct = np.unique(phot_cat[cosmos_id_name], return_counts=True)
        phot_cat["flux_scale"] = 1.0
        for dup_id in uniq[uniq_ct > 1]:
            print(f"Duplicate COSMOS ID : {dup_id}")
            total_flux = np.nansum(
                phot_cat[phot_cat[cosmos_id_name] == dup_id]["flux_auto"]
            )
            for idx in np.argwhere(phot_cat[cosmos_id_name] == dup_id):
                flux_scale = phot_cat["flux_auto"][idx] / total_flux
                for c in phot_cat.colnames[7:]:
                    if ("wn_" not in c) and (("_flux" in c) or ("_err" in c)):
                        phot_cat[c][idx] *= flux_scale
                phot_cat["flux_scale"][idx] = flux_scale

        phot_cat.write(
            passage_dir
            / field
            / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}.fits"
        )

        np.savetxt(
            passage_dir / field / f"{field}_filter_list_{fit_ver}_cosmos{cat_ver}.txt",
            filter_list,
            fmt="%s",
        )

    extcorr_path = (
        passage_dir
        / field
        / f"{field}_bagpipes_input_{fit_ver}_cosmos{cat_ver}_extcorr.fits"
    )
    try:
        extcorr_cat = Table.read(extcorr_path)
    except:
        extcorr_cat = apply_dust_correction(phot_cat, filter_list)
        extcorr_cat.write(extcorr_path)

    if config["general"].get("fit_emlines", False):
        bagpipes_emlines_path = (
            passage_dir
            / field
            / f"{field}_bagpipes_input_emlines_{fit_ver}_cosmos{cat_ver}.fits"
        )
        try:
            bagpipes_emlines_cat = Table.read(bagpipes_emlines_path)
        except:

            extcorr_cat = Table.read(extcorr_path)

            reformat_kwargs = dict(
                keep_ids=np.asarray(extcorr_cat["id_photcat"]),
                out_name=f"{field}_bagpipes_input_emlines_{fit_ver}_cosmos{cat_ver}.fits",
                out_dir=passage_dir / field,
                line_names=config["general"].get("line_names", DEFAULT_FIT_LINES),
            )
            if config["general"].get("emlines_is_grizli", True):
                print("Looking for grizli speccat")
                reformat_grizli_speccat(
                    passage_dir
                    / field
                    / config["catalogues"]
                    .get("emline_cat_name_template", "{field}_speccat.fits")
                    .format(field=field),
                    **reformat_kwargs,
                )
            else:
                print("Looking for linefinding speccat")
                reformat_lines_list(
                    passage_dir
                    / field
                    / config["catalogues"]
                    .get("emline_cat_name_template", "{field}lines_catalog_recon.dat")
                    .format(field=field),
                    **reformat_kwargs,
                )

            # print ()

    # exit()


LINEFINDING_TO_GRIZLI_NAMES_MAP = {
    "s2_6716_6731": "SII",
    "ha_6550_6565_6585": "Ha",
    "o3_4959_5007": "OIII",
    "hb_4863": "Hb",
    "o2_3727_3730": "OII",
    "ne3_3869": "NeIII-3867",
}

GRIZLI_TO_LINEFINDING_NAMES_MAP = {
    v: k for k, v in LINEFINDING_TO_GRIZLI_NAMES_MAP.items()
}

GRIZLI_TO_CLOUDY_MAP = {
    # Paschen series
    "PaA": {
        "cloudy": [
            "H  1  1.87510m",
        ],
        "wave": 18756.3,
    },
    "PaB": {
        "cloudy": [
            "H  1  1.28181m",
        ],
        "wave": 12821.7,
    },
    "PaG": {
        "cloudy": [
            "H  1  1.09381m",
        ],
        "wave": 10941.2,
    },
    "PaD": {
        "cloudy": [
            "H  1  1.00494m",
        ],
        "wave": 10052.2,
    },
    # Balmer Series
    "Ha": {
        "cloudy": [
            "H  1  6562.80A",
            "N  2  6583.45A",
            "N  2  6548.05A",
        ],
        "wave": 6564.697,
    },
    "Hb": {
        "cloudy": ["H  1  4861.32A"],
        "wave": 4862.738,
    },
    "Hg": {
        "cloudy": ["H  1  4340.46A"],
        "wave": 4341.731,
    },
    "Hd": {
        "cloudy": ["H  1  4101.73A"],
        "wave": 4102.936,
    },
    # Oxygen
    "OIII": {
        "cloudy": [
            "O  3  5006.84A",
            "O  3  4958.91A",
        ],
        "wave": 5008.240,
    },
    "OIII-4363": {
        "cloudy": ["O  3  4363.21A"],
        "wave": 4364.436,
    },
    "OII": {
        "cloudy": [
            "O  2  3726.03A",
            "O  2  3728.81A",
        ],
        "wave": 3728.48,
    },
    # Sulphur
    "SIII-9530": {
        "cloudy": [
            "S  3  9530.62A",
        ],
        "wave": 9530.62,
    },
    "SIII-9068": {
        "cloudy": [
            "S  3  9068.62A",
        ],
        "wave": 9068.62,
    },
    "SII": {
        "cloudy": [
            "S  2  6730.82A",
            "S  2  6716.44A",
        ],
        "wave": 6725.48,
    },
    "SIII-6314": {
        "cloudy": [
            "S  3  6312.06A",
        ],
        "wave": 6313.81,
    },
    # Helium
    "HeI-1083": {
        "cloudy": [
            "Blnd  1.08302m",
        ],
        "wave": 10830.3,
    },
    "HeI-5877": {
        "cloudy": [
            "Blnd  5875.66A",
        ],
        "wave": 5877.249,
    },
    "HeI-3889": {
        "cloudy": [
            "He 1  3888.64A",
        ],
        "wave": 3889.75,
    },
    # Neon
    "NeIII-3867": {
        "cloudy": [
            "Ne 3  3868.76A",
        ],
        "wave": 3869.87,
    },
}
DEFAULT_FIT_LINES = ["SII", "Ha", "OIII", "Hb", "OII", "NeIII-3867"]

GRIZLI_TO_CLOUDY_NAMES_ONLY = {k: v["cloudy"] for k, v in GRIZLI_TO_CLOUDY_MAP.items()}


def reformat_lines_list(
    orig_path: Path,
    keep_ids: ArrayLike,
    out_name: str | None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    line_names_map: dict = LINEFINDING_TO_GRIZLI_NAMES_MAP,
    line_names: list = DEFAULT_FIT_LINES,
) -> Path:
    """
    Reformat a linefinding `*.dat` catalogue as a `*.fits` catalogue.

    Parameters
    ----------
    orig_path : Path
        The location of the original linefinding catalogue.
    keep_ids : ArrayLike
        A list of IDs to keep in the reformatted catalogue.
    out_name : Path, optional
        The name of the output (reformatted) catalogue. If `None`,
        `"_reformat"` will be appended to the end of the original name.
    out_dir : Path | None, optional
        The directory in which the outputs will be saved. The default
        value of `None` means the catalogues will be saved to the same
        directory as the input.
    overwrite : bool, optional
        Overwrite the output catalogues if they exist already, by default
        `False`.
    line_names_map : dict, optional
        A dictionary mapping the names used by the linefinding code to
        those used by `grizli`, by default
        `LINEFINDING_TO_GRIZLI_NAMES_MAP`.
    line_names : list, optional
        A list of emission line names as used by `grizli`, by default
        `DEFAULT_FIT_LINES`. Only names in
        this list will be preserved in the output catalogue.

    Returns
    -------
    Path
        The location of the reformatted line list catalogue.
    """

    if out_dir is None:
        out_dir = orig_path.parent

    reformat_path = out_dir / f"{orig_path.stem}_reformat.fits"

    if out_name is None:
        out_name = f"{orig_path.stem}_reformat.fits"
    if (out_dir / out_name).is_file() and not overwrite:
        return out_dir / out_name
    else:

        orig_tab = Table.read(
            orig_path,
            format="ascii.csv",
            delimiter="\\s",
            comment="\\s*#",
        )
        orig_tab.write(out_dir / f"{orig_path.stem}.fits", overwrite=True)

        # Strip out any commented lines before writing reformatted table
        del orig_tab.meta["comments"]

        orig_tab["id_photcat"] = orig_tab["objid"].astype(int)

        orig_tab = orig_tab[np.isin(orig_tab["id_photcat"], keep_ids)]

        reformat_tab = orig_tab["id_photcat", "chisq", "fwhm", "fwhm_error"]

        for lf, g in line_names_map.items():
            if g not in line_names:
                continue
            reformat_tab[f"flux_{g}"] = orig_tab[f"{lf}_flux"]
            reformat_tab[f"err_{g}"] = orig_tab[f"{lf}_error"]

            not_in_filter = np.logical_not(
                check_coverage(
                    GRIZLI_TO_CLOUDY_MAP[g]["wave"] * (1 + orig_tab["redshift"])
                )
            )
            for s in ["flux", "err"]:
                reformat_tab[f"{s}_{g}"][not_in_filter] = np.nan

        reformat_tab.write(out_dir / out_name, overwrite=True)

        return out_dir / out_name


def reformat_grizli_speccat(
    orig_path: Path,
    keep_ids: ArrayLike,
    out_name: str | None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    line_names: list = DEFAULT_FIT_LINES,
) -> Path:
    """
    Reformat a grizli speccat to include only essential information.

    Parameters
    ----------
    orig_path : Path
        The location of the original grizli catalogue.
    keep_ids : ArrayLike
        A list of IDs to keep in the reformatted catalogue.
    out_name : Path, optional
        The name of the output (reformatted) catalogue. If `None`,
        `"_reformat"` will be appended to the end of the original name.
    out_dir : Path | None, optional
        The directory in which the outputs will be saved. The default
        value of `None` means the catalogues will be saved to the same
        directory as the input.
    overwrite : bool, optional
        Overwrite the output catalogues if they exist already, by default
        `False`.
    line_names : list, optional
        A list of emission line names as used by `grizli`, by default
        `DEFAULT_FIT_LINES`. Only names in
        this list will be preserved in the output catalogue.

    Returns
    -------
    Path
        The location of the reformatted catalogue.
    """

    if out_dir is None:
        out_dir = orig_path.parent

    reformat_path = out_dir / f"{orig_path.stem}_reformat.fits"

    if out_name is None:
        out_name = f"{orig_path.stem}_reformat.fits"
    if (out_dir / out_name).is_file() and not overwrite:
        return out_dir / out_name
    else:

        orig_tab = Table.read(orig_path)

        orig_tab["id_photcat"] = orig_tab["id"].astype(int)
        orig_tab = orig_tab[np.isin(orig_tab["id_photcat"], keep_ids)]

        reformat_tab = orig_tab["id_photcat", "chimin", "dof", "z_map"]

        for g in line_names:
            not_in_filter = np.logical_not(
                check_coverage(
                    GRIZLI_TO_CLOUDY_MAP[g]["wave"] * (1 + reformat_tab["z_map"])
                )
            )
            for s in ["flux", "err"]:
                reformat_tab[f"{s}_{g}"] = orig_tab[f"{s}_{g}"]
                reformat_tab[f"{s}_{g}"][not_in_filter] = np.nan
            # reformat_tab[f"err_{g}"] = orig_tab[f"err_{g}"]

        # for lf, g in line_names_map.items():
        #     reformat_tab[f"flux_{g}"] = orig_tab[f"{lf}_flux"]
        #     reformat_tab[f"err_{g}"] = orig_tab[f"{lf}_error"]

        # exit()

        reformat_tab.write(out_dir / out_name, overwrite=True)

        return out_dir / out_name


NIRISS_001_FILTER_LIMITS = {
    # "F090W": [7960, 10050],
    "F115W": [10010, 12930],
    "F150W": [13200, 16810],
    "F200W": [17380, 22420],
}


def check_coverage(
    obs_wavelength: float | ArrayLike, filter_limits: dict = NIRISS_001_FILTER_LIMITS
) -> bool | ArrayLike:
    """
    Check if a line is covered by the NIRISS filters.

    Parameters
    ----------
    obs_wavelength : float or ArrayLike
        The observed wavelength(s) of the line.
    filter_limits : dict, optional
        A dictionary, where the keys are the names of the grism filters,
        and the values are array-like, containing
        ``[min_wavelength, max_wavelength]``. Defaults to
        ``NIRISS_001_FILTER_LIMITS``.

    Returns
    -------
    bool or ArrayLike
        ``True`` if the line falls within the filter coverage, else
        ``False``.
    """

    obs_wavelength = np.atleast_1d(obs_wavelength)
    covered = np.zeros((len(filter_limits), *obs_wavelength.shape), dtype=bool)
    for i, (k, v) in enumerate(filter_limits.items()):
        covered[i] = (obs_wavelength >= v[0]) & (obs_wavelength <= v[-1])
    return np.bitwise_or.reduce(covered, axis=0)


def correct_pipes_params(
    current_dict: dict, list_keys: ArrayLike = ["bin_edges_low", "bin_edges_high"]
) -> dict:
    """
    Correct bagpipes fit instructions so that prior limits are tuple types.

    Parameters
    ----------
    current_dict : dict
        The dictionary to search through (recursively).
    list_keys : ArrayLike, optional
        These keys are retained as lists, by default
        `["bin_edges_low", "bin_edges_high"]`.

    Returns
    -------
    dict
        The corrected fit instructions.
    """
    for k, v in current_dict.items():
        if isinstance(v, dict):
            current_dict[k] = correct_pipes_params(v, list_keys)
        if isinstance(v, list) and np.logical_not(np.isin(k, list_keys)):
            current_dict[k] = tuple(v)
    return current_dict


if __name__ == "__main__":

    # reformat_lines_list(
    #     Path(
    #         "/media/sharedData/data/2026_02_14_passage-par028/v0.5_reduction/Par028_output_recon/Par028lines_catalog_recon.dat"
    #     )
    # )

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from project_2025c import plotting_scripts

    ref_cat = Table.read(
        # "/media/sharedData/data/2026_01_08__PASSAGE/ref_cats/passagepipe_v0.5_SED_fits_cosmosweb_v1.0.0-alpha.fits"
        "/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data/cats/SED_fits_v1.0.2_cosmos2020.fits"
    )
    new_cat = Table.read(
        "/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data/cats/SED_fits_v1.0.2_cosmosweb.fits"
    )

    # new_cat = new_cat[np.logical_not(new_cat["id_photcat"].mask)]

    # tab_name_1 = "v1.0.0"
    # tab_name_2 = "v1.0.2"
    tab_name_1 = "web"
    tab_name_2 = "2020"

    matched = join(
        ref_cat,
        new_cat,
        # keys_left="passage_id",
        # keys_right="id",
        # keys_left=["Par", "passage_id"],
        # keys_right=["field", "id_photcat"],
        keys = ["id_huberty"],
        table_names=[tab_name_1, tab_name_2],
    )

    print(matched.colnames)

    q = "stellar_mass"
    # matched = matched[matched[f"{q}_50_2020"] > 0]

    fig, axs = plt.subplots(
        1,
        1,
        constrained_layout=True,
        figsize=(plotting_scripts.aanda_columnwidth, 3),
    )

    ax = axs
    ax.errorbar(
        matched[f"{q}_50_{tab_name_1}"],
        matched[f"{q}_50_{tab_name_2}"],
        xerr=[
            matched[f"{q}_50_{tab_name_1}"] - matched[f"{q}_16_{tab_name_1}"],
            matched[f"{q}_84_{tab_name_1}"] - matched[f"{q}_50_{tab_name_1}"],
        ],
        yerr=[
            matched[f"{q}_50_{tab_name_2}"] - matched[f"{q}_16_{tab_name_2}"],
            matched[f"{q}_84_{tab_name_2}"] - matched[f"{q}_50_{tab_name_2}"],
        ],
        fmt=".",
        ecolor=(0.0, 0.0, 0.0, 0.5),
        markerfacecolor="none",
        zorder=-1,
    )
    ax.scatter(
        matched[f"{q}_50_{tab_name_1}"],
        matched[f"{q}_50_{tab_name_2}"],
        alpha=0.7,
        c="purple",
        s=10,
    )
    lims = np.asarray([ax.get_xlim(), ax.get_ylim()])
    lims = np.array([np.nanmin(lims), np.nanmax(lims)])
    print(lims)

    print(
        sigma_clipped_stats(
            matched[f"{q}_50_{tab_name_1}"] - matched[f"{q}_50_{tab_name_2}"]
        )
    )

    ax.plot(lims, lims, linestyle=":", c="k", alpha=0.7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    axs.set_xlabel(rf"$\log_{{10}}\left(M_*/M_{{\odot}}\right)$ ({tab_name_1})")
    axs.set_ylabel(rf"$\log_{{10}}\left(M_*/M_{{\odot}}\right)$ ({tab_name_2})")

    # plt.savefig(plot_dir / "compare_mass_2020_2025.pdf")

    plt.show()
