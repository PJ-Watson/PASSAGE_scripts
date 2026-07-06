#!/bin/bash

echo 'export CONF_DIRS="${HOME}/conf_dirs"' >> ~/.bashrc
echo 'export GRIZLI="${CONF_DIRS}/grizli"' >> ~/.bashrc
echo 'export iref="${GRIZLI}/iref"' >> ~/.bashrc
echo 'export jref="${GRIZLI}/jref"' >> ~/.bashrc
echo 'export CRDS_PATH="${CONF_DIRS}/crds_cache"' >> ~/.bashrc
echo 'export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"' >> ~/.bashrc

source ~/.bashrc

mkdir -p $CONF_DIRS
mkdir -p $GRIZLI/CONF
mkdir -p $GRIZLI/templates
mkdir -p $iref
mkdir -p $jref

python -c '
import grizli.utils
grizli.utils.fetch_default_calibs(get_acs=True)
grizli.utils.fetch_config_files(get_acs=True, get_jwst=True)
grizli.utils.symlink_templates(force=False)

import os
import grizli
import shutil
from pathlib import Path

os.chdir(os.path.join(grizli.GRIZLI_PATH, "CONF"))
if not os.path.exists("GR150C.F115W.221215.conf"):
    os.system("wget \"https://zenodo.org/record/7628094/files/niriss_config_221215.tar.gz?download=1\" -O niriss_config_221215.tar.gz")
    os.system("tar xzvf niriss_config_221215.tar.gz")

if not os.path.exists("niriss_sens_221215.tar.gz"):
    os.system("wget \"https://zenodo.org/record/7628094/files/niriss_sens_221215.tar.gz\" -O niriss_sens_221215.tar.gz")
    os.system("tar xzvf niriss_sens_221215.tar.gz")

if not os.path.exists("NIRISS_F200W_GR150R.V5.conf"):
    os.system("wget \"https://zenodo.org/records/10955821/files/npirzkal/NGDEEP_NIRISS_CALIB-v5.zip\" -O NGDEEP_NIRISS_CALIB-v5.zip")
    shutil.unpack_archive("NGDEEP_NIRISS_CALIB-v5.zip")
    for file in Path.cwd().glob("npirzkal-NGDEEP_NIRISS*/*"):
        file.rename(Path.cwd() / file.name)
'
