"""A tool to generate comparison plots between different catalogue versions."""

import argparse
import tomllib
from pathlib import Path

import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, join
from numpy.typing import ArrayLike

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
    "cat_ver_1",
    type=str,
    metavar="cat_ver_1",
    help="The identifying version string for the first catalogue.",
)
parser.add_argument(
    "cat_ver_2",
    type=str,
    metavar="cat_ver_2",
    help="The identifying version string for the second catalogue.",
)

parser.add_argument(
    "-q",
    type=str,
    metavar="quantity",
    default="stellar_mass",
    help="The quantity to plot, by default `stellar_mass`.",
)
parser.add_argument(
    "--min_bands",
    type=int,
    default=2,
    help="The minimum number of bands for the fit to be plotted.",
)
parser.add_argument(
    "--weighted",
    action=argparse.BooleanOptionalAction,
    help="Weight the best fit line by the uncertainties.",
)

parser.add_argument(
    "--plot_dir",
    type=str,
    help="The directory in which plots will be saved.",
)


def residual_estimator(
    params: lmfit.Parameters,
    x: ArrayLike,
    y: ArrayLike,
    x_err: ArrayLike | None = None,
    y_err: ArrayLike | None = None,
) -> ArrayLike:
    """
    The residual for a line fit with errors on x and y.

    Parameters
    ----------
    params : lmfit.Parameters
        The parameters for the best fit line.
    x : ArrayLike
        The x values of the data.
    y : ArrayLike
        The y values of the data.
    x_err : ArrayLike | None, optional
        The uncertainties on `x`, by default `None`.
    y_err : ArrayLike | None, optional
        The uncertainties on `x`, by default `None`.

    Returns
    -------
    ArrayLike
        The residuals of the fit.
    """

    a = params["a"].value
    b = params["b"].value
    x0 = params["x0"].value

    if (x_err is not None) and (y_err is not None):
        return (a * (x - x0) + b - y) / np.sqrt((a * x_err) ** 2 + (y_err) ** 2)


aanda_columnwidth = 256.0748 / 72.27
aanda_textwidth = 523.5307 / 72.27


def setup_aanda_style(dark: bool = False):
    """
    A helper function to setup the A&A style.

    Parameters
    ----------
    dark : bool, optional
        Use a dark plotting style, by default `False`.
    """

    rc_params = {
        "font.family": "serif",
        "font.size": 7,
        "figure.figsize": (aanda_columnwidth, 3),
        "text.usetex": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.labelsize": 7,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.labelsize": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 0.5,
        "lines.markersize": 3.5,
        "image.interpolation": "none",
        "text.latex.preamble": (
            r"""
        \usepackage{amsmath}
        \usepackage{txfonts}
        \usepackage{siunitx}
        %
        \DeclareMathAlphabet{\mathsc}{OT1}{cmr}{m}{sc}
        \def\testbx{bx}%
        \DeclareRobustCommand{\ion}[2]{%
        \relax\ifmmode
        \ifx\testbx\f@series
        {\mathbf{#1\,\mathsc{#2}}}\else
        {\mathrm{#1\,\mathsc{#2}}}\fi
        \else\textup{#1\,{\mdseries\textsc{#2}}}%
        \fi}
        %
        """
        ),
    }

    if dark:
        rc_params |= {
            "text.color": "white",
            "axes.facecolor": "0C1C23",  # axes background color
            "axes.edgecolor": "#e1e9ec",  # axes edge color
            "axes.labelcolor": "#e1e9ec",
            "grid.color": "#e1e9ec",
            "legend.edgecolor": "#e1e9ec",
            "legend.facecolor": "inherit",
            "legend.labelcolor": "#e1e9ec",
            "xtick.color": "#e1e9ec",
            "ytick.color": "#e1e9ec",
            "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
            "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
        }

    plt.rcdefaults()
    mpl.rcParams.update(rc_params)

    return


latex_names = {
    "stellar_mass": r"$\log_{{10}}\left(M_*/M_{{\odot}}\right)$",
    # "sfr" : r"$\log_{{10}}\left( {\rm{SFR}} / M_{{\odot}} / {\rm{yr}} \right)$"
    "ssfr": r"$\log_{{10}}\left( {\rm{sSFR}} / {\rm{yr}} \right)$",
}


if __name__ == "__main__":

    args = parser.parse_args()

    cat_dir = Path(args.cat_dir)
    if args.plot_dir is None:
        args.plot_dir = Path(cat_dir).parent / "plots"
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True, parents=True)

    cat_dir = Path(args.cat_dir)

    ref_cat = Table.read(cat_dir / f"SED_fits_{args.cat_ver_1}.fits")
    new_cat = Table.read(cat_dir / f"SED_fits_{args.cat_ver_2}.fits")

    tab_name_1 = args.cat_ver_1
    tab_name_2 = args.cat_ver_2

    matched = join(
        ref_cat,
        new_cat,
        keys=["id_huberty"],
        table_names=[tab_name_1, tab_name_2],
    )

    q = args.q

    matched = matched[
        np.logical_not(
            np.logical_or(
                matched[f"{q}_50_{tab_name_1}"].mask,
                matched[f"{q}_50_{tab_name_2}"].mask,
            )
        )
        & (matched[f"n_bands_{tab_name_1}"] >= args.min_bands)
        & (matched[f"n_bands_{tab_name_2}"] >= args.min_bands)
    ]
    # matched.pprint()
    # print (np.unique(matched[f"field_{tab_name_1}"]))

    x = matched[f"{q}_50_{tab_name_1}"]
    y = matched[f"{q}_50_{tab_name_2}"]
    xerr = [
        x - matched[f"{q}_16_{tab_name_1}"],
        matched[f"{q}_84_{tab_name_1}"] - x,
    ]
    yerr = [
        y - matched[f"{q}_16_{tab_name_2}"],
        matched[f"{q}_84_{tab_name_2}"] - y,
    ]

    # print(np.nanmedian(x[np.isfinite(x)]))
    params = lmfit.create_params(a=0, b=0, x0=dict(value=np.nanmedian(x), vary=False))

    out = lmfit.minimize(
        residual_estimator,
        params,
        kws={
            "y": y,
            "x": x,
            "y_err": np.nanmedian(yerr, axis=0) if args.weighted else np.ones_like(y),
            "x_err": np.nanmedian(xerr, axis=0) if args.weighted else np.ones_like(y),
        },
    )

    for dark, save_type in enumerate(["pdf", "svg"]):

        if dark:
            continue

        setup_aanda_style(dark)

        fig, axs = plt.subplots(
            1,
            1,
            constrained_layout=True,
            figsize=(aanda_columnwidth, 3),
        )

        ax = axs
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=".",
            ecolor=(0.0, 0.0, 0.0, 0.5),
            markerfacecolor="none",
            zorder=-1,
        )
        ax.scatter(
            x,
            y,
            alpha=0.7,
            c="purple",
            s=10,
        )
        lims = np.asarray([ax.get_xlim(), ax.get_ylim()])
        lims = np.array([np.nanmin(lims), np.nanmax(lims)])
        # print(lims)

        # print(sigma_clipped_stats(x - y))

        ax.plot(lims, lims, linestyle=":", c="k", alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        axs.set_xlabel(
            rf"{latex_names.get(q, rf"\texttt{{{q}}}")} [\texttt{{{tab_name_1}}}]"
        )
        axs.set_ylabel(
            rf"{latex_names.get(q, rf"\texttt{{{q}}}")} [\texttt{{{tab_name_2}}}]"
        )
        # axs.set_ylabel(rf"$\log_{{10}}\left(M_*/M_{{\odot}}\right)$ ({tab_name_2})")

        xlims = np.asarray(axs.get_xlim())

        axs.plot(
            xlims,
            (xlims - out.params["x0"].value) * out.params["a"].value
            + out.params["b"].value,
            c="red",
            linestyle="--",
        )
        axs.set_xlim(*xlims)

        axs.annotate(
            rf"$\noindent a={out.params["a"].value:.3f}\pm{out.params["a"].stderr:.3f}$"
            + "\n"
            + rf"$b={out.params["b"].value:.3f}\pm{out.params["b"].stderr:.3f}$"
            + "\n"
            + rf"$x_0={out.params["x0"].value:.3f}$",
            (0.1, 0.9),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )
        # print(dir(out.params["a"]))

        # out.params.pretty_print()
        # print(fit_report(out))

        plt.savefig(
            plot_dir / f"{q}_{tab_name_1}_{tab_name_2}_"
            f"min_bands_{args.min_bands}_{"weighted" if args.weighted else "unweighted"}.{save_type}"
        )

        plt.show()
