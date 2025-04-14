import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import combinations

from gpmap.linop import calc_vjs_variance_components


def calc_variance_components(f):
    vc = defaultdict(lambda: 0)
    marginal_sites = {}
    marginal_pw = {}
    total_variance = np.sum((f - f.mean()) ** 2)
    for k in range(1, 10):
        m_j = 3**k
        vjs_k = defaultdict(lambda: 0)
        site_k = defaultdict(lambda: 0)
        vjs = calc_vjs_variance_components(f, a=4, sl=9, k=k)

        for j, lambda_j in vjs.items():
            vc[k] += lambda_j * m_j / total_variance

            for site in j:
                site_k[site] += lambda_j * m_j / total_variance

            if k > 1:
                for a, b in combinations(j, 2):
                    vjs_k[(a, b)] += lambda_j * m_j / total_variance

        if k > 1:
            marginal_pw[k] = vjs_k
        marginal_sites[k] = site_k

    marginal_pw = pd.DataFrame(marginal_pw).reset_index()
    cols = list(range(2, 10))
    marginal_pw.columns = ["i", "j"] + cols
    marginal_pw["sum"] = marginal_pw[cols].sum(1)
    marginal_sites = pd.DataFrame(marginal_sites)
    vc = pd.DataFrame({"vc": pd.Series(vc)})

    return vc, marginal_sites, marginal_pw


if __name__ == "__main__":
    print("Loading MAP estimates")
    data = pd.read_csv("results/vcregression.map.csv", index_col=0)

    print("Computing variance components in VC regression MAP")
    vc, marginal_sites, marginal_pw = calc_variance_components(data["f"])

    print("Storing variance components calculations")
    fpath = "results/vcregression.map.variance_components.csv"
    vc.to_csv(fpath)

    fpath = "results/vcregression.map.pairwise_marginal_epistasis.csv"
    marginal_pw.to_csv(fpath)

    fpath = "results/vcregression.map.site_marginal_epistasis.csv"
    marginal_sites.to_csv(fpath)
