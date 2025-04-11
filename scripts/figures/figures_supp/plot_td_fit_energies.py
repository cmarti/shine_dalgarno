import numpy as np
import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import holoviews as hv
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp

import matplotlib.pyplot as plt
from gpmap.utils import read_edges

from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    hv.extension("matplotlib")

    nodes = pd.read_parquet("results/thermodynamic_model.nodes.pq")
    print(nodes)
    data = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    energies = data.drop("y_pred", axis=1).values

    p = np.exp(-energies - logsumexp(-energies, axis=1, keepdims=True))
    p0 = 1.0 / p.shape[1]
    max_entropy = -np.sum([p0 * np.log(p0)] * p.shape[1])
    entropy = -np.sum(p * np.log(p), axis=1)
    print(p.max())
    print(entropy.max(), max_entropy)

    fig, subplots = plt.subplots(1, 2)

    axes = subplots[0]
    axes.hist(
        entropy,
        bins=100,
        alpha=0.2,
        color="black",
        label="Neutral",
        density=True,
    )
    axes.set_xlabel("Entropy")
    axes.set_ylabel("Count")
    axes.hist(
        entropy,
        bins=100,
        alpha=0.2,
        color="grey",
        weights=nodes["stationary_freq"],
        label="Selection",
        density=True,
    )
    axes.set_xlabel("Entropy")
    axes.set_ylabel("Count")
    axes.legend(loc=1)

    axes = subplots[1]
    axes.scatter(x=entropy, y=data["y_pred"])
    axes.set_xlabel("Entropy")
    axes.set_ylabel("Predicted GFP")
    fig.savefig("figures/td_entropy_vs_gfp.png", dpi=300)

    exit()

    fig = sns.pairplot(
        energies.drop("y_pred", axis=1),
        kind="hist",
        # grid_kws={"bins": 100, "cmap": "magma"},
        # figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.8),
    )

    fig.savefig("figures/td_energies_hist.png", dpi=300)
