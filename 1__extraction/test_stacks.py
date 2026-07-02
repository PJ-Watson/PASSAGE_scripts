import numpy as np
from pathlib import Path
import tomllib

config_path = "/media/sharedData/python/py3.13_PIE/code/PASSAGE_scripts/1__extraction/config_files/CINECA_config_par682.toml"
# config_path = "/media/sharedData/python/py3.13_PIE/code/PASSAGE_scripts/1__extraction/config_lcs-par08.toml"


with open(config_path, "rb") as f:
    config = tomllib.load(f)

beam_kwargs = config["extraction"].get("beams", {})

# extractions_dir = Path(
#     "/media/sharedData/data/2026_05_20_parallel_fields/2.0.2/passage-par682/grizli_home/Extractions"
# )

# root_dir = Path(config["general"].get("root_dir", Path.cwd()))
root_dir = Path("/media/sharedData/data/2026_05_20_parallel_fields")
field = config["general"].get("field")

field_name = f"{config["general"].get("field_prefix")}-{field}".lower()

field_obs_IDs = config["general"].get("field_obs_ids")
proposal_IDs = config["general"].get("proposal_ids")

reduction_ver = config["general"].get("reduction_ver", "1.0.0")

reduction_dir = root_dir / reduction_ver / field_name
reduction_dir.mkdir(exist_ok=True, parents=True)

grizli_home_dir = reduction_dir / "grizli_home"
prep_dir = grizli_home_dir / "Prep"
extractions_dir = grizli_home_dir / "Extractions"

from grizli import multifit
from niriss_tools.grism.utils import gen_stacked_beams, _cluster_beams_centres

import warnings

# Quiet some of the common the grizli-induced warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm

for w in [VerifyWarning, FITSFixedWarning, UnitsWarning, RuntimeWarning]:
    warnings.simplefilter("ignore", category=w)


if __name__=="__main__":

    obj_id = 901
    # obj_id = 611
    import os

    os.chdir(extractions_dir)

    if not (extractions_dir
            # / "beams"
            / f"{field_name}_{obj_id:0>5}.beams.fits").is_file():
        mb = multifit.MultiBeam(
            str(
                extractions_dir
                / "beams"
                / f"{field_name}_{obj_id:0>5}.beams.fits"
            ),
            group_name=field_name,
            **beam_kwargs,
        )

        from sklearn.cluster import DBSCAN
        dbscan_kwargs = {"eps": 5, "min_samples" : 3}
        cluster_beams = False
        separate_fwcpos = True

        stacked_mb = gen_stacked_beams(mb, cluster_beams=cluster_beams, **beam_kwargs)
        stacked_mb.write_master_fits()

    from grizli.pipeline import auto_script
    from grizli import fitting

    args = auto_script.generate_fit_params(
        field_root=field_name,
        **beam_kwargs,
        **config["extraction"].get("fit_params", {}),
    )
    pline = args.get("pline", {})
    if config["extraction"].get("recalculate_size", True):
        pline["size"] = int(
            np.clip(2 * 50 * 0.06, a_min=3, a_max=30)
        )

    fitting.run_all_parallel(
        int(obj_id),
        pline=pline,
        get_output_data=True,
        verbose=True,
        zr =[ 3.1,3.4]
    )

    # for beam in mb.beams:
    #     print (beam.beam.fwcpos)

    # test_beam_fwcpos = mb.beams[0].compute_model(in_place=False, is_cgs=False).reshape(mb.beams[0].sh)
    # # print (test_beam.shape)
    # old = mb.beams[0].beam.direct
    # mb = multifit.MultiBeam(
    #     str(
    #         extractions_dir
    #         # / "backup"
    #         / "backup_fwcpos_none"
    #         / "beams"
    #         / f"{field_name}_{obj_id:0>5}.beams.fits"
    #     ),
    #     group_name=field_name,
    #     **beam_kwargs,
    # )
    # mb.beams[0].beam.direct = old

    # import matplotlib.pyplot as plt

    # print (f"{mb.beams[0].beam.fwcpos=}")
    # test_beam_none = mb.beams[0].compute_model(in_place=False, is_cgs=False).reshape(mb.beams[0].sh)

    # fig, axs = plt.subplots(3,1, sharex=True, sharey=True)

    # axs[0].imshow(test_beam_none)
    # axs[1].imshow(test_beam_fwcpos)
    # axs[2].imshow((test_beam_none-test_beam_fwcpos)/test_beam_fwcpos)
    # plt.show()
    # for filt, pa_info in tqdm(mb.PA.items(), desc="Stacking beams"):
    #     for pa, pa_beam_idxs in pa_info.items():

    #         pa_beam_idxs = np.array(pa_beam_idxs)
    #         fwcpos_arr = np.asarray([mb.beams[b].beam.fwcpos for b in pa_beam_idxs])
    #         print (np.unique(fwcpos_arr, return_index=True, return_inverse=True))

    #         if cluster_beams:

    #             if separate_fwcpos:
    #                 fwcpos_arr = np.asarray([mb.beams[b].beam.fwcpos for b in pa_beam_idxs])
    #                 grouped_beam_idxs = []

    #                 unique, u_inv = np.unique(fwcpos_arr, return_inverse=True)

    #                 for i, u in enumerate(unique):
    #                     # print (i, u)
    #                     # print (u_inv)
    #                     # print (u_inv==i)
    #                     grouped_beam_idxs.extend(_cluster_beams_centres(mb, pa_beam_idxs[u_inv==i], dbscan_kwargs))

    #             else:
    #                 # pa_beam_idxs = np.array(pa_beam_idxs)
    #                 grouped_beam_idxs = _cluster_beams_centres(mb, pa_beam_idxs, dbscan_kwargs)

    #         else:
    #             grouped_beam_idxs = [pa_beam_idxs]

    #         print (grouped_beam_idxs)

    #         exit()