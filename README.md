# NIRISS/WFSS Pipeline

This repository contains a number of scripts enabling the end-to-end reduction of NIRISS/WFSS data.

## Installation

A minimal installation environment can be found in `environment.yml`, to be used with any conda-like package manager (tested with `micromamba`). As of writing, this contained all of the necessary requirements to run through Stage 1 (reduction, extraction, and redshift fitting). After initialising the environment, it is also necessary to run `setup_vars.sh`, which adds the necessary environment variables to the bash shell, and downloads the latest trace/wavelength configuration files from several Zenodo repositories.

By default, the `grizli` and CRDS configuration files will be stored under the user's home directory. If an alternative location is preferred, modify line 3 of `setup_vars.sh`, specifically `CONF_DIRS="${HOME}/conf_dirs"` to point to the preferred directory.

## Stage 1: Extraction

This is split into three separate parts, with an additional script, `offline_steps.py`, that may be necessary on some clusters.

### Offline Processing

If (as is common on a number of HPC setups), the compute nodes themselves do not have external network access, two additional steps are necessary.

The first is to change a keyword in the TOML config, under `[general]`, setting `process_offline = true`. This will check if the required files are present, and end the processes if not. Much better to find out at the start than after getting halfway through reducing a field.

The second is to run the `offline_steps.py` script, passing the same configuration file as for the other steps. This queries CRDS for the necessary reference files and downloads them, as well as constructing a reference astrometric catalogue in order to align the direct imaging to an external reference.

### A: Reduction

This takes the `*uncal.fits` files from MAST, and returns the `*GrismFLT.fits` used by `grizli`.

### B: Fitting

This runs the beam extraction and redshift fitting. Recommended to run this using `mpirun` to take advantage of parallel processing.

### C: Forced Extraction (Optional)

TBD.

## Stage 2: Multiregion Processing


## Stage 3: SED Fitting
