from pathlib import Path

import numpy as np
from astropy.table import Table, join, vstack

field_name = "passage-par682"

extractions_dir = Path(
    "/media/watsonp/ArchivePJW/backup/data/2026_02_18_passage-par682/grizli_home/Extractions"
    # "/media/sharedData/data/INAFGDrive/2026_02_18_passage-par682/grizli_home/Extractions"
)

import warnings

# Quiet some of the common the grizli-induced warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astropy.wcs import FITSFixedWarning
from grizli.multifit import MultiBeam

for w in [VerifyWarning, FITSFixedWarning, UnitsWarning, RuntimeWarning]:
    warnings.simplefilter("ignore", category=w)

mb_kwargs = dict(fcontam=0.2, min_sens=0.0, min_mask=0.0)


def create_1d_hdul(beams_file):

    obj_id = int(beams_file.name.split("_")[1].split(".")[0])
    if (extractions_dir / "1D_RC" / f"{field_name}_{obj_id:0>5}.1D.fits").is_file():
        return

    mb = MultiBeam(str(beams_file), **mb_kwargs)

    new_hdul = mb.oned_spectrum_to_hdu()

    for k, v in mb.PA.items():
        for pa, beam_idx in v.items():
            try:
                _mb = MultiBeam([mb.beams[i] for i in beam_idx], **mb_kwargs)
                out = _mb.oned_spectrum_to_hdu()
                out[-1].header["EXTVER"] = pa
                out[-1].header["FILTER"] = _mb.beams[0].grism.filter
                new_hdul.append(out[-1])
            except:
                continue
            # print (out[0].header)
        # mb = MultiBeam()
        # print (v)
    # mb = MultiBeam()
    try:
        del _mb
    except:
        pass

    # new_hdul.info()

    new_hdul.writeto(extractions_dir / "1D_RC" / f"{field_name}_{mb.id:0>5}.1D.fits")
    del new_hdul
    del mb


from multiprocessing import Pool

beams_files = list((extractions_dir / "beams").glob("*.fits"))
beams_files.sort(reverse=True)

with Pool(processes=1) as pool:

    for beams_file in beams_files:

        pool.apply_async(create_1d_hdul, args=(beams_file,), error_callback=print)

    pool.close()
    pool.join()
