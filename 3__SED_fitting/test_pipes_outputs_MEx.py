import matplotlib.pyplot as plt

import numpy as np
from astropy.table import Table, join
from pathlib import Path

import plotutils
import sed_utils

plotutils.formatting.setup_aanda_style()

passage_dir = Path("/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data")


def juneau_2014(mass: np.ndarray, upper_curve: bool = True) -> np.ndarray:

    demarcation = np.zeros_like(mass)

    if upper_curve:
        low_idx = mass <= 10

        demarcation[low_idx] = 0.375 / (mass[low_idx] - 10.5) + 1.14
        x = mass[np.logical_not(low_idx)]
        demarcation[np.logical_not(low_idx)] = (
            410.24 - 109.333 * x + 9.71731 * x**2 - 0.288244 * x**3
        )

    return demarcation


if __name__ == "__main__":

    field_name = "Par034"
    fields = [
        # "Par682",
        # "Par016",
        # "Par019",
        # "Par021",
        # "Par034",
        # "Par040",
        # "Par042",
        # "Par044",
        # "Par2744",
        # "Par028"
        "Par023"
    ]
    version = "v1.2.1"
    version = "v0.2.11"
    fig, axs = plt.subplots(
        constrained_layout=True, figsize=(plotutils.formatting.aanda_textwidth, 5)
    )

    axs.set_xlabel(r"$\log_{10}\left(M_*/ \rm{M}_{\odot}\right)$")
    axs.set_ylabel(
        r"$ \log_{10} \left(\rm{SFR}_{100}\, /\, M_{\odot} \rm{yr}^{-1}\right)$"
    )

    # for version in ["v0.2.9", "v0.2.10", "v0.2.11"]:
    # for i, version in enumerate(["v0.2.10", "v0.2.11", "v0.2.12"]):
    # for i, version in enumerate(["v0.2.11", "v0.2.12"]):
    # for i, version in enumerate(["v0.2.12", "v0.2.13", "v0.2.14"]):

    # for i, version in enumerate(["v1.2.1","v1.3.0"]):
    # for i, version in enumerate(["v1.2.1", "v1.3.2"]):
    for i, version in enumerate(["v1.2.0", "v1.3.2"]):
        # for i, version in enumerate(["v1.3.2"]):
        # for i, version in enumerate(["v1.3.0","v1.3.1"]):
        # for i, version in enumerate(["v1.3.0","v1.3.2"]):
        # for i, version in enumerate(["v1.3.3", "v1.3.4"]):

        for j, field in enumerate(fields):

            # sed_fit_tab = Table.read(
            #     passage_dir / field / f"{field}_full_{version}.fits"
            # )
            sed_fit_tab = Table.read(
                passage_dir.parent / "to_upload" / version / f"SED_fits_{version}_cosmosweb.fits"
            )
            # tab_path = list((passage_dir / field).glob(f"{field}_full_{version}*.fits"))
            # # tab_path = list(
            # #     (passage_dir / "pipes" / "cats").glob(f"{field}_fit*{version}*.fits")
            # # )
            # tab_path.sort(reverse=True)
            # print(tab_path)
            # sed_fit_tab = Table.read(tab_path[0])

            # sed_fit_tab = sed_fit_tab[
            #     (sed_fit_tab["flux_OIII"]>1e-20) &
            #     (sed_fit_tab["flux_OII"]>1e-20)
            # ]
            sed_fit_tab = sed_fit_tab[sed_fit_tab["stellar_mass_50"] > 0]
            sed_fit_tab = sed_fit_tab[
                (sed_fit_tab["flux_OIII"] / sed_fit_tab["err_OIII"] > 3)
                & (sed_fit_tab["flux_Hb"] / sed_fit_tab["err_Hb"] > 3)
                & (sed_fit_tab["flux_Hb"] > 1e-20)
            ]

            # plt.hist(sed_fit_tab["nebular:logU_50"], bins=np.arange(-4.2, -0.8, 0.1))
            # plt.show()

            # axs.hist(sed_fit_tab["stellar_mass_50"], label=version, histtype="stepfilled", alpha=0.6, bins=np.arange(6.5,11.5,0.25), color=f"C{i}")
            # axs.hist(sed_fit_tab["stellar_mass_50"], histtype="step", bins=np.arange(6.5,11.5,0.25), color=f"C{i}")

            # print (np.nanmedian(sed_fit_tab["stellar_mass_50"]))
            # axs.axvline(np.nanmedian(sed_fit_tab["stellar_mass_50"]), color=f"C{i}")

            axs.scatter(
                sed_fit_tab["stellar_mass_50"],
                np.log10(sed_fit_tab["flux_OIII"] / sed_fit_tab["flux_Hb"]),
                # c=sed_fit_tab["nebular:logU_50"],
                # c=sed_fit_tab["redshift"],
                # cmap="plasma",
                alpha=0.7,
                label=f"{field}, {version}",
            )
            # print(sed_fit_tab[sed_fit_tab["stellar_mass_50"] < 5])

            xlims = np.array([6.0, 12.0])
            print(
                (
                    juneau_2014(sed_fit_tab["stellar_mass_50"])
                    > np.log10(sed_fit_tab["flux_OIII"] / sed_fit_tab["flux_Hb"])
                ).sum(),
                len(sed_fit_tab),
            )
            # if i == 0:
            #     axs.plot(xlims, xlims - 8.0, c="k", linestyle=":")
            if (i == 0) & (j == 0):
                plot_range = np.linspace(*xlims, 1000)
                axs.plot(
                    plot_range,
                    # np.log10(sfms_popesso_23(10**plot_range, 1.0)),
                    juneau_2014(plot_range),
                    c="b",
                    linestyle="--",
                    linewidth=1.0,
                    label="Juneau+14",
                )
                # axs.plot(
                #     plot_range,
                #     np.log10(sfms_popesso_23(10**plot_range, 3.0)),
                #     c="r",
                #     linestyle="--",
                #     linewidth=1.0,
                #     label="Popesso+23 ($z=3$)",
                # )
            axs.set_xlim(xlims)
            axs.set_ylim(ymin=-0.5)

            # # prev_tab = Table.read("/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/analysis_v9/binned_colour_Ha_sn_6/compiled_integrated_grizli.fits")
            # # sed_fit_tab["ID_NIRISS"] = sed_fit_tab["#ID"].astype(int)
            # # joined_tab = join(prev_tab, sed_fit_tab, join_type="left", keys_left=["ID_NIRISS"], keys_right=["ID_NIRISS"])
            # # axs.scatter(
            # #     joined_tab["stellar_mass_50"],
            # #     joined_tab["binned_stellar_mass"]+np.log10(joined_tab["magnification"])
            # # )
            # # axs.scatter(
            # #     np.log10(sed_fit_tab["flux_OIII"]/
            # #     sed_fit_tab["flux_OII"]),
            # #     sed_fit_tab["nebular:logU_50"]
            # # )

            # print (np.nanmedian(sed_fit_tab["ssfr_50"]))
            # axs.axvline(np.nanmedian(sed_fit_tab["ssfr_50"]), color=f"C{i}")

            # axs.hist(sed_fit_tab["ssfr_50"], label=version, histtype="stepfilled", alpha=0.6, bins=np.arange(-10,-7,0.1), color=f"C{i}")
            # axs.hist(sed_fit_tab["ssfr_50"], histtype="step", bins=np.arange(-10,-7,0.1), color=f"C{i}")
    axs.legend()
    # plt.savefig("HST_followup_v1.3.2.pdf")
    plt.show()
    exit()

    fig, axs = plt.subplots(constrained_layout=True)

    for field in fields:
        tab_path = list(
            (passage_dir / "pipes" / "cats").glob(f"{field}_fit*v0.2.12*.fits")
        )
        tab_path.sort(reverse=True)
        print(tab_path)
        sed_fit_tab_1 = Table.read(tab_path[0])
        tab_path = list(
            (passage_dir / "pipes" / "cats").glob(f"{field}_fit*v0.2.13*.fits")
        )
        tab_path.sort(reverse=True)
        print(tab_path)
        sed_fit_tab_2 = Table.read(tab_path[0])

        print(np.nanmedian(sed_fit_tab_1["chisq_phot"] - sed_fit_tab_2["chisq_phot"]))

        # plt.scatter(
        #     sed_fit_tab_2["ssfr_50"], sed_fit_tab_1["chisq_phot"] - sed_fit_tab_2["chisq_phot"]
        # )
        axs.scatter(
            sed_fit_tab_2["stellar_mass_50"] - sed_fit_tab_1["stellar_mass_50"],
            np.log10(sed_fit_tab_2["sfr_50"] / sed_fit_tab_1["sfr_50"]),
        )
        axs.axvline(0.0, c="k", linestyle=":")
        axs.axhline(0.0, c="k", linestyle=":")

        axs.set_xlabel(r"$\Delta\log_{10}\left(M_*/ \rm{M}_{\odot}\right)$")
        axs.set_ylabel(
            r"$\Delta \log_{10} \left(\rm{SFR}_{100}\, /\, M_{\odot} \rm{yr}^{-1}\right)$"
        )

    plt.show()
    exit()
    for field in ["Par2744"]:

        # sed_fit_tab = Table.read(passage_dir / field / f"{field}_full_{version}.fits")
        tab_path = list((passage_dir / "pipes" / "cats").glob(f"{field}_fit*.fits"))
        tab_path = list(
            (passage_dir / "pipes" / "cats").glob(f"{field}_fit*{version}*.fits")
        )
        tab_path.sort(reverse=True)
        print(tab_path[0])
        sed_fit_tab = Table.read(tab_path[0])

        # sed_fit_tab = sed_fit_tab[
        #     (sed_fit_tab["flux_OIII"]>1e-20) &
        #     (sed_fit_tab["flux_OII"]>1e-20)
        # ]
        sed_fit_tab = sed_fit_tab[sed_fit_tab["stellar_mass_50"] > 0]

        # plt.hist(sed_fit_tab["nebular:logU_50"], bins=np.arange(-4.2, -0.8, 0.1))
        # plt.show()

        # axs.scatter(
        #     sed_fit_tab["stellar_mass_50"],
        #     np.log10(sed_fit_tab["sfr_50"]),
        #     # c=sed_fit_tab["nebular:logU_50"],
        #     # c=sed_fit_tab["redshift"],
        #     # cmap="plasma"
        # )

        # Force extractions
        new_id_mapping = {
            10017: [17, 19, 21],
            10521: [521, 561],
            11689: [1689, 1718],
            2051: [2051, 2081],
            2166: [2166, 2172],
            12250: [2250, 2258, 2282, 2290],
            12355: [2355, 2363],
            2534: [2534, 2543],
            2762: [2762, 2781],
            2845: [2845, 2876],
            3384: [3384],
            3495: [3495],
            3838: [3838],
            3842: [3842],
            3881: [3881],
        }

        prev_tab = Table.read(
            "/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/analysis_v9/binned_colour_Ha_sn_6/compiled_integrated_grizli.fits"
        )
        old_ids = []
        for k, v in new_id_mapping.items():
            old_ids.extend(v)
        prev_tab = prev_tab[
            np.logical_not(np.isin(prev_tab["ID_NIRISS"], new_id_mapping.keys()))
            & np.logical_not(np.isin(prev_tab["ID_NIRISS"], old_ids))
        ]
        sed_fit_tab["ID_NIRISS"] = sed_fit_tab["#ID"].astype(int)
        joined_tab = join(
            prev_tab,
            sed_fit_tab,
            join_type="right",
            keys_left=["ID_NIRISS"],
            keys_right=["ID_NIRISS"],
        )

        # axs.scatter(
        #     joined_tab["binned_stellar_mass"] + np.log10(joined_tab["magnification"]),
        #     np.log10(joined_tab["binned_sfr"]) + np.log10(joined_tab["magnification"]),
        #     label="W26",
        #     # c=sed_fit_tab["nebular:logU_50"],
        #     # c=sed_fit_tab["redshift"],
        #     # cmap="plasma"
        # )

        axs.hist(
            joined_tab["binned_stellar_mass"] + np.log10(joined_tab["magnification"]),
            label="W26",
            histtype="stepfilled",
            alpha=0.6,
            bins=np.arange(6.5, 11.5, 0.25),
            color=f"C{i+1}",
        )
        axs.hist(
            joined_tab["binned_stellar_mass"] + np.log10(joined_tab["magnification"]),
            histtype="step",
            bins=np.arange(6.5, 11.5, 0.25),
            color=f"C{i+1}",
        )
        axs.legend()
        plt.show()
        exit()

        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs[0].errorbar(
            joined_tab["binned_stellar_mass"] + np.log10(joined_tab["magnification"]),
            joined_tab["stellar_mass_50"],
            xerr=np.abs(joined_tab["binned_stellar_mass_err"]),
            yerr=[
                joined_tab["stellar_mass_50"] - joined_tab["stellar_mass_16"],
                joined_tab["stellar_mass_84"] - joined_tab["stellar_mass_50"],
            ],
            fmt=".",
        )
        axs[0].scatter(
            joined_tab["binned_stellar_mass"] + np.log10(joined_tab["magnification"]),
            joined_tab["stellar_mass_50"],
        )

        axs[1].errorbar(
            np.log10(joined_tab["binned_sfr"]) + np.log10(joined_tab["magnification"]),
            np.log10(joined_tab["sfr_50"]),
            xerr=np.log10(joined_tab["binned_sfr"] + joined_tab["binned_sfr_err"])
            - np.log10(joined_tab["binned_sfr"]),
            yerr=[
                np.log10(joined_tab["sfr_50"]) - np.log10(joined_tab["sfr_16"]),
                np.log10(joined_tab["sfr_84"]) - np.log10(joined_tab["sfr_50"]),
            ],
            fmt=".",
        )
        axs[1].scatter(
            np.log10(joined_tab["binned_sfr"]) + np.log10(joined_tab["magnification"]),
            np.log10(joined_tab["sfr_50"]),
        )
        for a in axs:
            xlims = a.get_xlim()
            ylims = a.get_ylim()
            new_lims = np.array([min(xlims[0], ylims[0]), max(xlims[-1], ylims[-1])])
            a.plot(new_lims, new_lims, c="k", linestyle=":")
            a.set_xlim(new_lims)
            a.set_ylim(new_lims)
        # axs.scatter(
        #     prev_tab["binned_stellar_mass"]+np.log10(prev_tab["magnification"]),
        #     np.log10(prev_tab["binned_sfr"])+np.log10(prev_tab["magnification"]),
        #     # c=sed_fit_tab["nebular:logU_50"],
        #     # c=sed_fit_tab["redshift"],
        #     # cmap="plasma"
        # )
        # axs.scatter(
        #     np.log10(sed_fit_tab["flux_OIII"]/
        #     sed_fit_tab["flux_OII"]),
        #     sed_fit_tab["nebular:logU_50"]
        # )

    plt.show()
    exit()

    fig, axs = plt.subplots(constrained_layout=True)

    for path in ["SED_fits_v1.2.0_cosmosweb.fits", "SED_fits_v1.0.2_cosmosweb.fits"]:

        sed_fit_tab = Table.read(passage_dir / "cats" / path)
        axs.scatter(
            sed_fit_tab["stellar_mass_50"],
            np.log10(sed_fit_tab["sfr_50"]),
            # sed_fit_tab["ssfr_50"],
            # np.log10(sed_fit_tab["sfr_50"]),
            # sed_fit_tab["nebular:logU_50"],
            s=3,
            # cmap="plasma"
        )
        axs.set_xlabel("sSFR")
        axs.set_ylabel("logU")
        print(
            (
                (sed_fit_tab["nebular:logU_50"] > -1.5)
                & (sed_fit_tab["ssfr_50"] > -8.3)
            ).sum()
            / len(sed_fit_tab)
        )

    ref_cat = Table.read(
        "/media/sharedData/data/2026_01_08__PASSAGE/ref_cats/COSMOSWeb_mastercatalog_v1.1.fits",
        "CIGALE",
    )
    ref_cat["cosmoswebid_1"] = np.arange(len(ref_cat))

    matched_cat = join(
        ref_cat,
        sed_fit_tab,
        join_type="right",
        keys_left="cosmoswebid_1",
        keys_right="cosmoswebid_1",
        table_names=["cigale", "bagpipes"],
    )
    axs.scatter(
        np.log10(matched_cat["mass"]),
        np.log10(matched_cat["sfr_100myr"]),
    )

    fig, axs = plt.subplots(constrained_layout=True)
    axs.scatter(
        np.log10(matched_cat["sfr_100myr"]) - np.log10(matched_cat["mass"]),
        np.log10(sed_fit_tab["sfr_50"]) - sed_fit_tab["stellar_mass_50"],
    )

    plt.show()
