import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from itertools import combinations
from scipy.stats import pearsonr
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    print("Loading data")
    fpath = "results/vcregression.map.site_marginal_epistasis.csv"
    marginal_sites = pd.read_csv(fpath, index_col=0)
    marginal_sites.index = [str(x) for x in range(-13, -4)]
    marginal_sites = marginal_sites.T.iloc[::-1, :]

    sites1 = ["-13", "-12", "-11", "-10"]
    sites2 = ["-9", "-8", "-7"]
    sites3 = ["-6", "-5"]

    df = pd.DataFrame(
        {
            "-13-10": marginal_sites[sites1].mean(1),
            "-9-7": marginal_sites[sites2].mean(1),
            "-6-5": marginal_sites[sites3].mean(1),
        }
    )
    print("Average percentage of variance explained by interactions of every possible order involving each site")
    print(df)

    fpath = "results/vcregression.map.pairwise_marginal_epistasis.csv"
    marginal_pw = pd.read_csv(fpath, index_col=0)
    marginal_pw["d<3"] = [
        np.abs(i - j) < 3 for i, j in marginal_pw[["site1", "site2"]].values
    ]
    marginal_pw["d<4"] = [
        np.abs(i - j) < 4 for i, j in marginal_pw[["site2", "site1"]].values
    ]
    d3 = marginal_pw.groupby("d<3")["variance_pw_perc"].mean()
    print("Average percentage of pairwise variance explained by pairs of sites")
    print("\twithin 3 nucleotides: {:.2f}".format(d3.values[1]))
    print("\tbeyond 3 nucleotides: {:.2f}".format(d3.values[0]))

    d4 = marginal_pw.groupby("d<4")["variance_high_order_perc"].mean()
    print("Average percentage of higher order variance explained by pairs of sites")
    print("\twithin 4 nucleotides: {:.2f}".format(d4.values[1]))
    print("\tbeyond 4 nucleotides: {:.2f}".format(d4.values[0]))

    s1, s2 = list(range(4)), [4, 5, 6]
    sites1 = list(combinations(s1, 2))
    sites2 = list(combinations(s2, 2))

    v1 = marginal_pw.set_index(["site1", "site2"]).loc[sites1, "variance_pw_perc"].mean()
    v2 = marginal_pw.set_index(["site1", "site2"]).loc[sites2, "variance_high_order_perc"].mean()

    print('Average percentage of pairwise variance explained by pairs of sites {}: {:.2f}'.format(s1, v1))
    print('Average Percentage of higher order variance explained by pairs of sites {}: {:.2f}'.format(s2, v2))
