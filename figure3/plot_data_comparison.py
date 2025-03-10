import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    seqdeft = pd.read_csv("results/seqdeft.full.csv", index_col=0)
    vcregression = pd.read_csv("results/vcregression.full.csv", index_col=0)

    x = np.log10(seqdeft["Q_star"]).values
    y = vcregression["y"].values
    print(x)
    print(y)

    fig, axes = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH * 0.33, FIG_WIDTH * 0.275),
    )
    r = spearmanr(x, y)[0]
    lims = (-8.5, -2)
    xticks = np.arange(-8, -2)
    xticklabels = [
        "$10^{-8}$",
        "$10^{-7}$",
        "$10^{-6}$",
        "$10^{-5}$",
        "$10^{-4}$",
        "$10^{-3}$",
    ]
    axes.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=0.5)
    sns.histplot(
        x=x,
        y=y,
        cmap="inferno",
        ax=axes,
        bins=100,
        cbar=True,
        cbar_kws={"label": "Number of sequences"},
    )
    axes.set(
        ylabel="VC regression log(GFP)",
        xlabel="SeqDEFT sequence probability",
        xlim=lims,
        xticks=xticks,
    )
    axes.set_xticklabels(xticklabels, rotation=45)
    axes.text(
        0.05,
        0.95,
        r"Spearman $\rho$" + "={:.2f}".format(r),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig("figures/seqdeft_vcregression_comparison.png", dpi=300)
