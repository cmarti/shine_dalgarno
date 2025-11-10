import numpy as np
import pandas as pd

from gpmap.summary import GPmapSummarizer

if __name__ == "__main__":
    print("Loading MAP estimates")
    data = pd.read_csv("results/vcregression.map.csv", index_col=0)
    sd_map = GPmapSummarizer(4, 9, f=data["f"].values)

    rmsme = sd_map.calc_root_mean_squared_epistatic_coeff(P=1)
    rmsec = sd_map.calc_root_mean_squared_epistatic_coeff(P=2)
    print("Root mean squared epistatic coefficient {:.2f}".format(rmsec))
    print("Root mean squared mutational effect {:.2f}".format(rmsme))

    print("Computing V_k variance components in VC regression MAP")
    v_k_vcs = sd_map.calc_V_k_variance_components()
    v_k_vcs.to_csv("results/vcregression.map.variance_components.csv")

    print("Computing V_U variance components in VC regression MAP")
    v_u_vcs = sd_map.calc_V_U_variance_components()

    print("Computing site-marginal variance components in VC regression MAP")
    sites = sd_map.calc_sites_variance_perc(v_u_vcs).T
    sites.to_csv("results/vcregression.map.site_marginal_epistasis.csv")

    print("Computing site-pairs variance components in VC regression MAP")
    pairs = sd_map.calc_site_pairs_variance_perc(v_u_vcs)
    pw_v_u_vcs = v_u_vcs.loc[v_u_vcs["k"] == 2, :]
    pairs_pw = sd_map.calc_site_pairs_variance_perc(pw_v_u_vcs)
    pairs_high_order = sd_map.calc_site_pairs_variance_perc(v_u_vcs, min_k=3)
    pairs["variance_pw"] = pairs_pw["variance"]
    pairs["variance_pw_perc"] = pairs_pw["variance_perc"]
    pairs["variance_high_order"] = pairs_high_order["variance"]
    pairs["variance_high_order_perc"] = pairs_high_order["variance_perc"]
    # Ensure calculations match
    assert np.allclose(
        pairs["variance_pw"] + pairs["variance_high_order"], pairs["variance"]
    )
    pairs.to_csv("results/vcregression.map.pairwise_marginal_epistasis.csv")
