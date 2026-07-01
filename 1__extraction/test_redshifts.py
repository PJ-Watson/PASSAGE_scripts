import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.rotations import Rotation2D
from astropy.table import Table, join, vstack
from grizli.grismconf import GRIZLI_PATH, JwstDispersionTransform, load_grism_config

# test_config = load_grism_config("CONF/CUSTOM/COMBINE_NGDEEP_A_GRIZLI_{1}_{0}_V1.conf".format("GR150C","F200W"))
test_config = load_grism_config(
    os.path.join(
        GRIZLI_PATH, "CONF", "NIRISS_F200W_GR150C.V5.conf".format("GR150C", "F200W")
    )
)

test_config.get_beams()

# test_config.get_beam_trace()
centre = np.array([500, 500])

dx = np.arange(-100, 300)
# print (test_config.dxlam)
dx = test_config.dxlam["A"]

# print (dir(test_config))

fig, axs = plt.subplots(constrained_layout=True)

axs.scatter(*centre)

# fwcpos_arr = np.arange(354.1, 354.40, 0.0)
fwcpos_arr = np.array([-0.2, 0.0, 0.2])

for fwcpos in fwcpos_arr:

    axs.plot(
        centre[0] + dx,
        centre[1]
        + test_config.get_beam_trace(
            x=centre[0], y=centre[1], dx=dx, fwcpos=fwcpos + 354.2111
        )[0],
        # cmap="plasma",
        # c=
        label=rf"$\Delta\mathrm{{fwcpos}}={fwcpos}$",
    )

    # axs.plot(
    #     dx * -1,
    #     -1
    #     * test_config.get_beam_trace(x=centre[0], y=centre[1], dx=dx, fwcpos=fwcpos)[0],
    # )

axs.set_xlabel("x (pixels)")
axs.set_ylabel("y (pixels)")
# axs.plot(dx, test_config.get_beam_trace(dx=dx)[0])
axs.legend()
plt.show()

# print (dir(JwstDispersionTransform))

# xs = np.array([0,1,2,3,3,3])
# ys = np.array([0,0,0,0,1,2])
# angle = 90


# axs.plot(xs, ys)
# rotation = Rotation2D(angle)

# axs.plot(*JwstDispersionTransform.rotate_coordinates(xs, ys, np.radians(angle), [0,0]))

# axs.plot(*rotation(xs, ys))

# plt.show()

exit()


field_name = "passage-par034"

extractions_dir = Path(
    "/media/watsonp/ArchivePJW/backup/data/2026_05_20_parallel_fields/2.0.0/passage-par034/grizli_home/Extractions"
)

phot_cat = Table.read(extractions_dir / f"{field_name}_phot.fits")

# phot_cat = phot_cat[np.isin(phot_cat["id"], candidate_obj_ids)]
# phot_cat = phot_cat[phot_cat["mag_auto"] < mag_limit]

old_table = Table.read(extractions_dir / "Par034_speccat_cwt.fits")

import astropy.units as u
from astropy.coordinates import SkyCoord

old_coords = SkyCoord(ra=old_table["ra"], dec=old_table["dec"], unit="deg")
new_coords = SkyCoord(ra=phot_cat["ra"], dec=phot_cat["dec"], unit="deg")
out = new_coords.match_to_catalog_sky(old_coords)
idx, d2d, d3d = new_coords.match_to_catalog_sky(old_coords)
max_sep = 0.3 * u.arcsec
sep_constraint = d2d < max_sep
phot_cat[sep_constraint].pprint()
# print (out)
# # print (old_table)
# print (len(np.unique(phot_cat[sep_constraint]["id"])))
# exit()
phot_cat = phot_cat[sep_constraint]
phot_cat["v1_id"] = old_table[idx[sep_constraint]]["id"]

from multiprocessing import Pool

new_tab_data = []

with Pool() as pool:

    for row_file in (extractions_dir / "row").glob("*row.fits"):
        pool.apply_async(Table.read, args=(row_file,), callback=new_tab_data.append)
    pool.close()
    pool.join()

new_tab = vstack(new_tab_data)
new_tab.pprint()

new_tab = join(new_tab, phot_cat["id", "v1_id"], keys="id", join_type="left")
new_tab.write(extractions_dir / f"{field_name}_speccat_PJW.fits")

matched_cat = join(
    old_table,
    new_tab,
    join_type="left",
    keys_left="id",
    keys_right="v1_id",
    table_names=["v1", "PJW"],
)

# matched_cat = matched_cat


zeroth_order_ids = [
    465,
    502,
    516,
    529,
    670,
    699,
    595,
    724,
    776,
    1011,
    1102,
    1358,
    1498,
    1535,
    1589,
    1682,
    1709,
    1713,
    1755,
    1780,
    1818,
    1828,
    1829,
    2057,
    2152,
    2175,
    2232,
    2334,
    2558,
    2588,
    2593,
    2681,
    2684,
    2687,
    2762,
    3006,
    3023,
    3604,
    3764,
    4132,
    4141,
    4288,
    4310,
    4348,
    4362,
    4396,
    4407,
    4452,
    4461,
    4615,
    4717,
    4729,
    4730,
    4778,
    4782,
]


matched_cat = matched_cat[
    (matched_cat["zwidth1_PJW"] < 0.01) & (matched_cat["zwidth1_v1"] < 0.01)
]
matched_cat.pprint()
print(np.nansum(matched_cat["zwidth1_PJW"] > 0))

import matplotlib.pyplot as plt

print(matched_cat.colnames)

fig, axs = plt.subplots()

axs = np.atleast_1d(axs)
axs[0].scatter(
    matched_cat["redshift_v1"],
    matched_cat["redshift_PJW"],
    c=np.isin(matched_cat["v1_id"], zeroth_order_ids),
)
axs[0].set_xlabel("Redshift v1")
axs[0].set_ylabel("Redshift PJW")
plt.show()

fig, axs = plt.subplots()

delta = (matched_cat["redshift_v1"] - matched_cat["redshift_PJW"]) / (
    1 + matched_cat["redshift_v1"]
)

axs = np.atleast_1d(axs)
axs[0].scatter(
    matched_cat["redshift_v1"],
    delta,
    c=matched_cat["zwidth1_PJW"],
    vmax=0.05,
    # c=np.isin(matched_cat["v1_id"], zeroth_order_ids),
)
axs[0].set_xlabel("Redshift v1")
axs[0].set_ylabel("dz/1+z")
plt.show()

# phot_cat = join
# phot_cat.write(
#     extractions_dir / "matched_Par034_speccat_cwt.fits", overwrite=True
# )
