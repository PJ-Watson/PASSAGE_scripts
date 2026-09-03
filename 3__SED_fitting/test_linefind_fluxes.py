import matplotlib.pyplot as plt

import numpy as np
from astropy.table import Table
from pathlib import Path

import plotutils
import sed_utils

plotutils.formatting.setup_aanda_style()

passage_dir = Path("/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data")


if __name__ == "__main__":

    field_name = "Par034"
    fields = [
        "Par682",
        # "Par016",
        # "Par019",
        # "Par021",
        # "Par034",
        # "Par040",
        # "Par042",
        # "Par044",
    ]

    # flux_names = ["f115w_flux_auto", "f150w_flux_auto", "f200w_flux_auto"]
    fig, axes = plt.subplots(
        len(fields),
        # len(flux_names),
        1,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        # figsize=(8, 3*len(fields)),
    )
    for j, field_name in enumerate(fields):

        # axs = axes[j]

        linefinding_cat = Table.read(
            passage_dir / field_name / f"{field_name}lines_catalog_farhan.fits"
        )
        linefinding_cat.sort("objid")
        grizli_cat = Table.read(
            Path(
                "/media/sharedData/data/2026_02_18_passage-par682/"
                "grizli_home/catalogues/passage-par682_compiled_grizli_v1.0.2.fits"
            )
        )

        grizli_cat = grizli_cat[np.isin(grizli_cat["id"], linefinding_cat["objid"])]
        grizli_cat.sort("id")

        redshift_idx = (
            (
                (grizli_cat["redshift"] - linefinding_cat["redshift"])
                / (1 + grizli_cat["redshift"])
            )
            < 1e-3
        ) #& (linefinding_cat["n2_6585_flux"] > 0)

        grizli_cat = grizli_cat[redshift_idx]
        linefinding_cat = linefinding_cat[redshift_idx]

        # test_id = 849
        # grizli_cat = grizli_cat[grizli_cat["id"]==test_id]
        # linefinding_cat = linefinding_cat[linefinding_cat["objid"]==test_id]

        grizli_cat.pprint()
        linefinding_cat.pprint()

        # plt.hist(
        #     # linefinding_cat["ha_6565_flux"] / linefinding_cat["n2_6550_flux"],
        #     # linefinding_cat["ha_6565_flux"] / linefinding_cat["n2_6585_flux"],
        #     linefinding_cat["n2_6550_flux"] / linefinding_cat["n2_6585_flux"],
        #     # linefinding_cat["n2_6585_flux"],
        #     # np.arange(29,31,0.1)
        #     # np.arange(0,1e-11,1e-12)
        # )
        # plt.show()
        # exit()

        # for gr, lf in sed_utils.GRIZLI_TO_LINEFINDING_NAMES_MAP.items():
        #     # plt.scatter()
        #     if gr == "OIII":
        #         plt.scatter(
        #             np.log10(grizli_cat[f"flux_{gr}"]),
        #             np.log10(linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"]),
        #             # grizli_cat[f"flux_{gr}"],linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"],
        #             # c=(grizli_cat["redshift"] - linefinding_cat["redshift"])
        #             # / (1 + grizli_cat["redshift"]),
        #         )
        #         # plt.scatter(
        #         #     np.log10(grizli_cat[f"flux_{gr}"]),
        #         #     np.log10(
        #         #         (
        #         #             linefinding_cat[f"{lf}_flux"]
        #         #             + 3 * linefinding_cat["n2_6550_flux"]
        #         #         )
        #         #         / grizli_cat[f"flux_{gr}"]
        #         #     ),
        #         #     # grizli_cat[f"flux_{gr}"],linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"],
        #         #     # c=(grizli_cat["redshift"] - linefinding_cat["redshift"])
        #         #     # / (1 + grizli_cat["redshift"]),
        #         # )
        #         # print(
        #         #     np.nanmedian(
        #         #         np.log10(
        #         #             (
        #         #                 linefinding_cat[f"{lf}_flux"]
        #         #                 + 3 * linefinding_cat["n2_6550_flux"]
        #         #             )
        #         #             / grizli_cat[f"flux_{gr}"]
        #         #         )
        #         #     )
        #         # )
        #         # print(
        #         #     np.nanmedian(
        #         #         np.log10(
        #         #             linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"]
        #         #         ),
        #         #     )
        #         # )
        #     # plt.xlim(-1e-16, 1e-15)
        #     # plt.ylim(-1,1)
        #     # plt.xlim(-1e-16, 1e-15)
        #     plt.ylabel("log(linefinding/grizli)")
        #     plt.xlabel("log(grizli)")
        #     plt.ylim(-1, 1)

        plt.scatter(
            np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_OII"]),
            np.log10(
                np.log10(
                    linefinding_cat[f"o3_4959_5007_flux"]
                    / linefinding_cat[f"o2_3727_3730_flux"]
                )
                / np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_OII"])
            ),
            # grizli_cat[f"flux_{gr}"],linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"],
            # c=(grizli_cat["redshift"] - linefinding_cat["redshift"])
            # / (1 + grizli_cat["redshift"]),
        )
        q = np.log10(
                np.log10(
                    linefinding_cat[f"o3_4959_5007_flux"]
                    / linefinding_cat[f"o2_3727_3730_flux"]
                )
                / np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_OII"])
            )
        print (10**np.nanmedian(q))


        # plt.scatter(
        #     np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_Ha"]),
        #     np.log10(
        #         np.log10(
        #             linefinding_cat[f"o3_4959_5007_flux"]
        #             / linefinding_cat[f"ha_6550_6565_6585_flux"]
        #         )
        #         / np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_Ha"])
        #     ),
        #     # grizli_cat[f"flux_{gr}"],linefinding_cat[f"{lf}_flux"] / grizli_cat[f"flux_{gr}"],
        #     # c=(grizli_cat["redshift"] - linefinding_cat["redshift"])
        #     # / (1 + grizli_cat["redshift"]),
        # )
        # q = np.log10(
        #         np.log10(
        #             linefinding_cat[f"o3_4959_5007_flux"]
        #             / linefinding_cat[f"ha_6550_6565_6585_flux"]
        #         )
        #         / np.log10(grizli_cat[f"flux_OIII"] / grizli_cat[f"flux_Ha"])
        #     )
        # print (10**np.nanmedian(q))
        # photcat_hst = Table.read(
        #     passage_dir / field_name / f"{field_name}_photcat_hst.fits"
        # )

        # flux_names = ["f115w_flux_iso", "f150w_flux_iso", "f200w_flux_iso"]

        # for i, f in enumerate(flux_names):
        #     try:
        #         axs[i].scatter(
        #             # photcat[f.replace("flux", "mag")],
        #             np.log10(photcat[f]),
        #             np.log10(photcat_hst[f] / photcat[f]),
        #             c="k",
        #             alpha=0.3
        #         )
        #     except:
        #         pass
        #     if i == 0:
        #         axs[i].set_ylabel(f"log10([photcat_hst]/[photcat])")
        #     axs[i].set_xlabel(f"log10({f} [photcat])")
        #     if i==2:
        #         axs[i].annotate(field_name,(.95,0.95), xycoords="axes fraction", ha="right", va="top")

        # fig.suptitle(field_name)
    # plt.savefig("/home/watsonp/wrong_auto_fluxes_2.pdf")
    plt.show()

    # fig, axes = plt.subplots(
    #     len(fields),
    #     len(flux_names),
    #     sharex=True,
    #     sharey=True,
    #     constrained_layout=True,
    #     figsize=(8, 3 * len(fields)),
    # )
    # for j, field_name in enumerate(fields):

    #     axs = axes[j]

    #     photcat = Table.read(passage_dir / field_name / f"{field_name}_photcat.fits")
    #     photcat_hst = Table.read(
    #         passage_dir / field_name / f"{field_name}_photcat_hst.fits"
    #     )

    #     # flux_names = ["f115w_flux_iso", "f150w_flux_iso", "f200w_flux_iso"]

    #     for i, f in enumerate(flux_names):
    #         try:
    #             # axs[i].scatter(
    #             #     # photcat[f.replace("flux", "mag")],
    #             #     np.log10(photcat[f]),
    #             #     np.log10(photcat_hst[f] / photcat[f]),
    #             #     c="k",
    #             #     alpha=0.3
    #             # )
    #             axs[i].scatter(
    #                 # photcat[f.replace("flux", "mag")],
    #                 # np.log10(photcat[f]/photcat[f.replace("flux", "fluxerr")]),
    #                 # np.log10(photcat_hst[f]/photcat_hst[f.replace("flux", "fluxerr")]),
    #                 photcat[f] / photcat[f.replace("flux", "fluxerr")],
    #                 photcat_hst[f] / photcat_hst[f.replace("flux", "fluxerr")],
    #                 # np.log10(photcat_hst[f] / photcat[f]),
    #                 c="k",
    #                 alpha=0.3,
    #             )
    #         except:
    #             pass
    #         if i == 0:
    #             axs[i].set_ylabel(f"SN [photcat_hst]")
    #         if i == 2:
    #             axs[i].annotate(
    #                 field_name,
    #                 (0.95, 0.95),
    #                 xycoords="axes fraction",
    #                 ha="right",
    #                 va="top",
    #             )

    #         if j == len(fields) - 1:

    #             axs[i].set_xlabel(f"SN [photcat])")

    #         axs[i].semilogx()
    #         axs[i].semilogy()

    #         min_val = 5e-5
    #         max_val = 5e4

    #         axs[i].plot(
    #             [min_val, max_val],
    #             [min_val, max_val],
    #             c="red",
    #             linestyle=":",
    #             zorder=-1,
    #         )

    #         axs[i].set_xlim(min_val, max_val)
    #         axs[i].set_ylim(min_val, max_val)

    #         # fig.suptitle(field_name)
    # plt.savefig("/home/watsonp/photcat_hst_SN_comparison.pdf")
    # plt.show()
