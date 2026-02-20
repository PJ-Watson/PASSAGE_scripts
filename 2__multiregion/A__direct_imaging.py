"""An example workflow for aligning and reprojecting imaging data."""

import os
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

# from project_2025c.constants import *

# megascience_dir = root_dir / "archival" / "grizli-v2" / "JwstMosaics" / "v7"

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1467.pmap"
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = "CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf"

# https://github.com/PJ-Watson/niriss-tools
from niriss_tools import pipeline

# Change this to reduce a different field
root_dir = Path(os.getenv("ROOT_DIR"))
# root_dir = Path("/media/watsonp/ArchivePJW/backup/data")
field = "Par028"
date = "2026_02_14"

field_name = f"passage-{field.lower()}"

try:
    passage_tab = Table.read(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")
except:
    import pandas as pd

    df = pd.read_csv(root_dir / "JWST PASSAGE Cycle 1 - Cy1 Executed.csv")
    df = df[["Par#", "Obs Date", "PID", "Obs ID", "Filter", "Mode"]]
    df = df.ffill()
    df = df[~df["Obs Date"].str.contains("SKIPPED")]
    # print (df)
    passage_tab = Table.from_pandas(df)
    passage_tab.write(root_dir / "PASSAGE_obs_details.csv", format="ascii.csv")

# exit()

field_obs_IDs = list(passage_tab[passage_tab["Par#"] == field]["Obs ID"])
proposal_ID = list(passage_tab[passage_tab["Par#"] == field]["PID"])[0]

reduction_dir = Path(root_dir) / f"{date}_{field_name}"
reduction_dir.mkdir(exist_ok=True, parents=True)


def gen_psf_from_direct(
    grizli_dir: os.PathLike,
    filters: list[str] = ["f115wn-clear", "f150wn-clear", "f200wn-clear"],
    psf_size: int = 350,
) -> dict:
    """
    Generate a PSF aligned with the drizzled direct images.

    The PSF matches the rotation of the direct imaging using the ``"PA_V3"``
    header keyword.

    Parameters
    ----------
    grizli_dir : os.PathLike
        The directory containing the grizli folders ("Prep", "Extractions", etc.).
    filters : list[str]
        The names of the NIRISS filters to search for.
    psf_size : int
        The pixel size of the PSF to generate, by default `350`.

    Returns
    -------
    dict
        A dictionary with keys corresponding to each unique grism and filter
        combination, and values of the PSF image.
    """

    import cv2
    import stpsf
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs import utils as wcs_utils
    from drizzlepac.astrodrizzle import ablot, adrizzle
    from grizli import utils as grizli_utils

    psf_dir = grizli_dir / "Prep" / "psfs"
    psf_dir.mkdir(exist_ok=True, parents=True)

    # for output_filter in ["F115W", "F150W", "F200W"]:
    for output_filter in filters:
        if (psf_dir / f"psf_{output_filter.split("-")[0].lower()}_norm.fits").is_file():
            continue
        with fits.open(
            grizli_dir / "Prep" / f"{field_name}-{output_filter}_drc_sci.fits"
        ) as ref_hdul:

            for i, rate_filepath in enumerate(
                (grizli_dir / "Prep").glob("*_rate.fits")
            ):
                # print (rate_filepath)
                print(output_filter.split("-")[0].upper())
                hdr = fits.getheader(rate_filepath)
                if hdr["FILTER"] != "CLEAR":
                    continue
                if hdr["PUPIL"] != output_filter.split("-")[0].upper().removesuffix(
                    "N"
                ):
                    continue

                # if i > 6:
                #     continue
                with fits.open(rate_filepath) as rate_hdul:

                    print(rate_hdul[0].header["PA_V3"])
                    # exit()

                    if not (psf_dir / f"psf_{output_filter}.fits").is_file():

                        inst = stpsf.setup_sim_to_match_file(str(rate_filepath))
                        inst.pixelscale = 0.03
                        psf = inst.calc_psf(
                            fov_pixels=psf_size + 1,
                            # fov_arcsec=5,
                            # oversample=3,
                            display=False,
                            # fov_pixels=np.nanmax(beam_wcs._naxis) * 2 + 1,
                        )
                        for ext in range(len(psf)):

                            print(
                                f"Extension {ext} has oversampling factor = ",
                                psf[ext].header["OVERSAMP"],
                                f"\tPixelscale = {psf[ext].header['PIXELSCL']:.4f} arcsec/pix",
                                f"\tFWHM = {stpsf.measure_fwhm(psf, ext=ext):.4f} arcsec",
                            )
                        psf_data = psf["DET_DIST"].data

                        # psf_fake_img = np.zeros_like(rate_hdul["SCI"].data)
                        # # psf_fake_img.shape // 2
                        # # psf_fake_img = np.zeros((10,10))
                        # # psf_data = np.ones((3,3))
                        # h_idx = int((psf_fake_img.shape[0] - psf_data.shape[0]) / 2)  # + 1
                        # v_idx = int((psf_fake_img.shape[1] - psf_data.shape[1]) / 2)  # + 1
                        # psf_fake_img[
                        #     h_idx : h_idx + psf_data.shape[0],
                        #     v_idx : v_idx + psf_data.shape[1],
                        # ] = psf_data.astype(rate_hdul["SCI"].data.dtype)
                        # plt.imshow(np.log10(psf_fake_img))
                        # plt.show()

                        from scipy.ndimage import rotate

                        psf_fake_img = rotate(
                            psf_data[:-1, :-1],
                            angle=-rate_hdul[0].header["PA_V3"],
                            reshape=False,
                        )[100:-100, 100:-100]

                        psf_fake_img /= psf_fake_img.sum()

                        # psf_hdul = fits.HDUList(
                        psf_hdul = fits.HDUList(rate_hdul[0])
                        # fits.PrimaryHDU(),
                        psf_hdul.append(
                            fits.ImageHDU(
                                data=psf_fake_img,
                                # header=
                                name="PSF",
                            )
                        )
                        psf_hdul[1].header["PIXSCALE"] = 0.03

                        psf_hdul.writeto(
                            psf_dir
                            / f"psf_{output_filter.split("-")[0].lower()}_norm.fits",
                            overwrite=True,
                        )


if __name__ == "__main__":

    grizli_dir = reduction_dir / "grizli_home"

    prelim_dict_path = reduction_dir / "existing_photometry.yaml"

    try:
        # Reload the full obs dict with PSF info
        with open(prelim_dict_path, "r") as file:
            prelim_dict = yaml.safe_load(file)

    except:

        gen_psf_from_direct(reduction_dir / "grizli_home")

        prelim_dict = {}

        for filt in ["f115w", "f150w", "f200w"]:

            prelim_dict[f"jwst-niriss-{filt}"] = dict(
                filt=filt.upper(),
                instrument="NIRISS",
                telescope="JWST",
                sci=str(
                    list(
                        (grizli_dir / "Prep").glob(f"{field_name}*{filt}*drc_sci.fits")
                    )[0]
                ),
                var=str(
                    list(
                        (grizli_dir / "Prep").glob(f"{field_name}*{filt}*drc_var.fits")
                    )[0]
                ),
                psf=str(grizli_dir / "Prep" / "psfs" / f"psf_{filt}n_norm.fits"),
            )

        with open(prelim_dict_path, "w") as file:
            yaml.dump(prelim_dict, file, sort_keys=False)

    # Where to save the details of the PSF-matched images
    conv_dict_path = reduction_dir / "PSF_matched_photometry.yaml"

    # Where to save the reprojected and convolved images
    conv_out_dir = reduction_dir / "PSF_matched_photometry"
    conv_out_dir.mkdir(exist_ok=True, parents=True)

    # The reference mosaic to align the images to
    ref_mosaic = (
        reduction_dir / "grizli_home" / f"Prep" / f"{field_name}-ir_drc_sci.fits"
    )

    os.chdir(conv_out_dir)

    # Avoid parsing files again if the details already exist
    if conv_dict_path.is_file():

        with open(conv_dict_path, "r") as file:
            conv_dict = yaml.safe_load(file)

    else:

        from niriss_tools.isophotal import reproject_and_convolve

        conv_dict = {}
        for filt_key, old_details in prelim_dict.items():
            conv_dict[filt_key] = {
                k: v
                for k, v in old_details.items()
                if k not in ["exp", "sci", "var", "wht"]
            }

            for t in ["sci", "var"]:
                print(t, prelim_dict[filt_key][t])
                _conv_out_path = reproject_and_convolve(
                    ref_path=ref_mosaic,
                    orig_images=Path(prelim_dict[filt_key][t]),
                    psfs=Path(prelim_dict[filt_key]["psf"]),
                    psf_target=prelim_dict["jwst-niriss-f200w"]["psf"],
                    out_dir=conv_out_dir,
                    new_names=f"repr_{filt_key}_{t}.fits",
                    reproject_image_kw={
                        "method": "adaptive",
                        "compress": False,
                    },
                    new_wcs_kw={"resolution": 0.03},
                )

                conv_dict[filt_key][t] = str(_conv_out_path[0])

        with open(conv_dict_path, "w") as outfile:
            yaml.dump(conv_dict, outfile, default_flow_style=False, sort_keys=False)

    print(conv_dict)
