import numpy as np
from bagpipes.fitting.prior import dirichlet, prior
from bagpipes import utils
import matplotlib.pyplot as plt


class TestSFH:

    def __init__(self, log_sampling=0.0025):

        self.redshift = 0.01

        self.hubble_time = utils.age_at_z[utils.z_array == 0.0][0]

        # Set up the age sampling for internal SFH calculations.
        log_age_max = np.log10(self.hubble_time) + 9.0 + 2 * log_sampling
        self.ages = np.arange(6.0, log_age_max, log_sampling)
        self.age_lhs = utils.make_bins(self.ages, make_rhs=True)[0]
        self.ages = 10**self.ages
        self.age_lhs = 10**self.age_lhs
        self.age_lhs[0] = 0.0
        self.age_lhs[-1] = 10**9 * self.hubble_time
        self.age_widths = self.age_lhs[1:] - self.age_lhs[:-1]

        self.age_of_universe = 10**9 * np.interp(
            self.redshift, utils.z_array, utils.age_at_z
        )

    def delayed(self, sfr, param):

        age = param["age"] * 10**9
        tau = param["tau"] * 10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = t * np.exp(-t / tau)

    def pjw(self, sfr, param):
        tx = param["tx"]
        iyer_param = np.hstack([10.0, np.log10(param["sfr"]), len(tx), tx])
        iyer_sfh, iyer_times = db.tuple_to_sfh(iyer_param, self.redshift)
        iyer_ages = self.age_of_universe - iyer_times[::-1] * 10**9

        mask = self.ages < self.age_of_universe
        sfr[mask] = np.interp(self.ages[mask], iyer_ages, iyer_sfh[::-1])

        bin_edges = np.array(param["bin_edges"])[::-1] * 10**6
        bin_widths = np.diff(np.array(param["bin_edges"])[::-1] * 10**6)

        n_bins = len(bin_edges) - 1
        dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

        for i in range(1, n_bins + 1):
            mask = (self.ages < bin_edges[i - 1]) & (self.ages > bin_edges[i])
            sfr[mask] += 10 ** np.sum(dsfrs[: i - 1])

    def continuity(self, sfr, param):
        bin_edges = np.array(param["bin_edges"])[::-1] * 10**6
        n_bins = len(bin_edges) - 1
        dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

        for i in range(1, n_bins + 1):
            mask = (self.ages < bin_edges[i - 1]) & (self.ages > bin_edges[i])
            sfr[mask] += 10 ** np.sum(dsfrs[: i - 1])


if __name__ == "__main__":

    sfh = TestSFH()

    fig, axs = plt.subplots()

    n_bins = 7

    prior_obj = prior(
        [(-10, 10) for i in np.arange(n_bins - 1)],
        ["student_t" for i in np.arange(n_bins - 1)],
        [[] for i in np.arange(n_bins - 1)],
    )

    print(prior_obj.sample())

    for i in np.arange(0, 500):

        # rng = np.random.default_rng(i)

        sfr = np.zeros_like(sfh.ages)

        # param = dict(
        #     age = rng.uniform()*10,
        #     tau = rng.uniform()*1
        # )

        # sfh.delayed(sfr, param)

        dsfrs = prior_obj.sample()
        bins_edges = [0]
        bins_edges.extend(np.geomspace(3e7, sfh.ages[-1], int(n_bins - 1)) / 1e6)
        param = dict(
            # bin_edges=np.concatenate(np.array([0]), np.geomspace(30, sfh.ages[-1], int(n_bins-1))) / 1e6,
            bin_edges=bins_edges
        )
        for i in np.arange(1, n_bins):
            param[f"dsfr{i}"] = dsfrs[i - 1]

        print(param)

        sfh.continuity(sfr, param)
        # sfr /= np.nansum(sfr)
        # axs.plot(sfh.ages, sfr)
        total = np.sum(np.cumsum(sfr * sfh.age_widths))
        # sfr /= total
        axs.plot(
            np.log10(sfh.ages),
            np.log10(sfr / (total - np.cumsum(sfr * sfh.age_widths))),
            c="k",
            alpha=0.1,
        )

        # axs.plot(sfh.ages, np.log10(sfr/((np.cumsum(sfr)*sfh.age_widths)[::-1])))
        # axs.plot(sfh.ages, np.log10(sfr/((np.cumsum(sfr)*sfh.age_widths))))

        # axs.plot(sfh.ages, sfr/np.cumsum(sfr)*np.concatenate(np.array([0]), np.diff(sfh.ages)))
        # axs.plot(sfh.ages, sfr)

    plt.show()
