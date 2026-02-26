

A description of the SED fits for the PASSAGE in COSMOS fields.

Each release will include fits using both the COSMOS2020 and COSMOS-WEB catalogues, as the NIRISS fields covered by these catalogues do not fully overlap.


# Data Products


## Catalogues
 - The main output will be the SED fitting catalogues, available in Box under `{fit_version}/SED_fits_{fit_version}_cosmos{cat_version}.fits`.
 - Including the `{fit_version}` (e.g. `v1.0.2`, `v1.1.0`), and `{cat_version}` (e.g. `web`, `2020`) in the file name should minimise conflicts if you're comparing across multiple versions of each, but I would recommend only using a single catalogue for any given science case.
 - Alongside the catalogues, there will be a zipped folder named `{fit_version}/full_dir_archive_{fit_version}_cosmos{cat_version}.zip`.
 - These folders contain summary plots for each fitted galaxy in the corresponding catalogue, showing:
   - The observed photometry (blue points), model photometry (orange points), and model spectrum (shaded orange spectrum), in muJy. This is plotted as a function of observed wavelength (bottom) and rest-frame wavelength (top).
   - The SFH, showing the logSFR against the age of the galaxy (on a log scale). The corresponding redshift is plotted on the top x-axis.
   - Histograms of the posterior distribution for the fitted stellar mass, and the derived SFR (integrated over 100 Myr).

## Code
 - The code used for these SED fits is available at https://github.com/PJ-Watson/PASSAGE_scripts.
 - Following v1.0.2, each release will have a dedicated config file, and can be run using the same script (`run_fits.py`), e.g.:
   - `mpirun -n 8 python run_fits.py config_v1.2.0.toml`
 - Additional package requirements include:
   - dustmaps: https://dustmaps.readthedocs.io/en/latest/
     - By default, extinction corrections are performed using the Schlegel, Finkbeiner & Davis dust map.
   - bagpipes: https://bagpipes.readthedocs.io/en/latest/
     - Please read the installation instructions carefully, particularly regarding MultiNest installation.
   - bagpipes-extended: https://github.com/PJ-Watson/bagpipes-extended
     - Includes the redshift-varying continuity SFH (`contvz`).
     - Other additions include combining emission lines to account for the NIRISS resolution, and a number of requested columns added to the output catalogue.
   - mpi4py: https://mpi4py.readthedocs.io/en/stable/
     - Optional, but highly recommended to make the most of having more than one CPU core.
 - For any questions or bugs relating to the code, please get in touch!

## Complete Directory
 - TBD. I will attempt to zip or otherwise upload all bagpipes input and output files, to make it easier for anyone wanting to re-run the stellar mass fits (e.g. changing the redshift, or parameterisation).

# Roadmap

Copied from my slides in Bern:
 - ~~v1.0.1 is a bugfix release, correcting the uncertainties on the NIRISS flux_auto measurements.~~
 - v1.1.0 will include emission line fluxes as measured by grizli (TBD, the current speccat releases used the wrong redshift for flag 1.5 objects).
 - v1.2.0 will include emission line fluxes from careful continuum fitting (by KN and FH).

# Versions

## v1.0.2:
 - The "basic" version of the catalogue. For many use cases, this is adequate, and is likely the most comparable to other stellar mass catalogues in the literature.
 - Fits to the COSMOS-Web catalogue use all available bands, except IRAC Ch1-4. These are much lower sensitivity than many of the other observations, and a similar wavelength range is covered by JWST/NIRCam and MIRI.
 - Fits to the COSMOS2020 catalogue include the IRAC bands, to provide some constraints on the rest-frame IR flux, as well as GALEX NUV/FUV where available.
 - Fits are not performed if there is no match to a COSMOS ID.
 - SED fitting was performed using Bagpipes (Carnall+18).
 - We assume a 7 age bin continuity SFH (Leja+19), with the youngest fixed to (0,30) Myr, and the oldest to (age_of_universe - 500, age_of_universe). The intermediate age bins are logarithmically spaced.
 - We allow the metallicity and ionisation parameter to vary between (0.0,3.0) Z_sol and (-3.5,-1.0) respectively.
 - Dust attenuation uses the parameterisation of Calzetti+00, with A_V varying between (0, 3).

## v1.0.1:
 - Not to be used, if even downloaded. Catalogue IDs were incorrectly matched, and extinction corrections not applied.

## v1.0.0:
 - The version included in the original (arXiv v1) submitted version of Huberty+26.
 - This version is kept as a historical reference only, and any scientific use cases should be updated to the v1.0.2 release at a minimum.
 - The uncertainties on the NIRISS `auto` fluxes were discovered to be incorrect in the original v0.5 reduction, and so were scaled using the S/N in the 1.5" aperture fluxes as a reference.


# Changelog

## 2026-02-26:
 - Uploaded v1.0.2 COSMOS2020 catalogue.
 - Updated README with more details of the fitting procedure, and a description of all the available data products.

## 2026-02-25:
 - Uploaded v1.0.2 COSMOS-Web catalogue.
 - Updated README to clarify changes between catalogue versions.

## 2026-02-24:
 - Uploaded v1.0.1 catalogue.
 - Discovered bug in v1.0.1 catalogue.
 - Deleted v1.0.1 catalogue.
