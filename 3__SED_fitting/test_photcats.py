import matplotlib.pyplot as plt

import numpy as np
from astropy.table import Table
from pathlib import Path

import plotutils

plotutils.formatting.setup_aanda_style()

passage_dir = Path("/media/sharedData/data/2026_01_08__PASSAGE/PASSAGE_data")


if __name__ == "__main__":

    field_name = "Par034"
    fields = [
        # "Par682",
        "Par016",
        "Par019",
        "Par021",
        "Par034",
        # "Par040",
        # "Par042",
        # "Par044",
    ]
    flux_names = ["f115w_flux_auto", "f150w_flux_auto", "f200w_flux_auto"]
    # fig, axes = plt.subplots(
    #     len(fields),
    #     len(flux_names),
    #     sharex=True,
    #     sharey=True,
    #     constrained_layout=True,
    #     figsize=(8, 3*len(fields)),
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
    #             axs[i].scatter(
    #                 # photcat[f.replace("flux", "mag")],
    #                 np.log10(photcat[f]),
    #                 np.log10(photcat_hst[f] / photcat[f]),
    #                 c="k",
    #                 alpha=0.3
    #             )
    #         except:
    #             pass
    #         if i == 0:
    #             axs[i].set_ylabel(f"log10([photcat_hst]/[photcat])")
    #         axs[i].set_xlabel(f"log10({f} [photcat])")
    #         if i==2:
    #             axs[i].annotate(field_name,(.95,0.95), xycoords="axes fraction", ha="right", va="top")

    #         # fig.suptitle(field_name)
    # plt.savefig("/home/watsonp/wrong_auto_fluxes_2.pdf")
    # plt.show()

    fig, axes = plt.subplots(
        len(fields),
        len(flux_names),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        figsize=(8, 3 * len(fields)),
    )
    for j, field_name in enumerate(fields):

        axs = axes[j]

        photcat = Table.read(passage_dir / field_name / f"{field_name}_photcat.fits")
        photcat_hst = Table.read(
            passage_dir / field_name / f"{field_name}_photcat_hst.fits"
        )

        # flux_names = ["f115w_flux_iso", "f150w_flux_iso", "f200w_flux_iso"]

        for i, f in enumerate(flux_names):
            try:
                # axs[i].scatter(
                #     # photcat[f.replace("flux", "mag")],
                #     np.log10(photcat[f]),
                #     np.log10(photcat_hst[f] / photcat[f]),
                #     c="k",
                #     alpha=0.3
                # )
                axs[i].scatter(
                    # photcat[f.replace("flux", "mag")],
                    # np.log10(photcat[f]/photcat[f.replace("flux", "fluxerr")]),
                    # np.log10(photcat_hst[f]/photcat_hst[f.replace("flux", "fluxerr")]),
                    photcat[f] / photcat[f.replace("flux", "fluxerr")],
                    photcat_hst[f] / photcat_hst[f.replace("flux", "fluxerr")],
                    # np.log10(photcat_hst[f] / photcat[f]),
                    c="k",
                    alpha=0.3,
                )
            except:
                pass
            if i == 0:
                axs[i].set_ylabel(f"SN [photcat_hst]")
            if i == 2:
                axs[i].annotate(
                    field_name,
                    (0.95, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                )

            if j == len(fields) - 1:

                axs[i].set_xlabel(f"SN [photcat])")

            axs[i].semilogx()
            axs[i].semilogy()

            min_val = 5e-5
            max_val = 5e4

            axs[i].plot(
                [min_val, max_val],
                [min_val, max_val],
                c="red",
                linestyle=":",
                zorder=-1,
            )

            axs[i].set_xlim(min_val, max_val)
            axs[i].set_ylim(min_val, max_val)

            # fig.suptitle(field_name)
    plt.savefig("/home/watsonp/photcat_hst_SN_comparison.pdf")
    plt.show()
