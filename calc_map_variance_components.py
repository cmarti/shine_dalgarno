import numpy as np
import pandas as pd
from gpmap.src.randwalk import WMWalk
from gpmap.src.linop import calc_vjs_variance_components
from collections import defaultdict
from itertools import combinations

def calc_variance_components(f):
    vc = defaultdict(lambda: 0)
    marginal_sites = {}
    marginal_pw = {}
    total_variance = np.sum((f - f.mean()) ** 2)
    for k in range(1, 10):
        m_j = 3**k
        vjs_k = defaultdict(lambda: 0)
        site_k = defaultdict(lambda: 0)
        vjs = calc_vjs_variance_components(f, a=4, l=9, k=k)

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
    vc = pd.DataFrame({'vc': pd.Series(vc)})

    return vc, marginal_sites, marginal_pw


if __name__ == "__main__":
    print('Loading MAP estimates')
    vcregression = pd.read_csv("results/vcregression.full.csv", index_col=0)
    seqdeft = pd.read_csv("results/seqdeft.full.csv", index_col=0)
    
    print('Computing variance components in VC regression MAP')
    vc, marginal_sites, marginal_pw = calc_variance_components(vcregression['y'])
    vc.to_csv("results/vcregression.map_variance_components.csv")
    marginal_pw.to_csv("results/vcregression.map_pairwise_marginal_epistasis.csv")
    marginal_sites.to_csv("results/vcregression.map_site_marginal_epistasis.csv")

    print('Computing variance components in SeqDEFT MAP')
    vc, marginal_sites, marginal_pw = calc_variance_components(np.log10(seqdeft["Q_star"]))
    vc.to_csv("results/seqdeft.map_variance_components.csv")
    marginal_pw.to_csv("results/seqdeft.map_pairwise_marginal_epistasis.csv")
    marginal_sites.to_csv("results/seqdeft.map_site_marginal_epistasis.csv")
