import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    e_coli = pd.read_csv("data/seqdeft_inference.csv", index_col=0)
    b_sub = pd.read_csv("results/b_sub.seqdeft_inference.csv", index_col=0)

    x = np.log10(e_coli["Q_star"])
    y = np.log10(b_sub["Q_star"])

    fig, axes = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH * 0.35, FIG_WIDTH * 0.275),
    )
    r = spearmanr(x, y)[0]
    lims = (-10.5, -2)
    xticks = np.arange(-10, -2)
    xticklabels = [
        "$10^{-10}$",
        "$10^{-9}$",
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
        cbar_kws={"label": "Number of sequences", "shrink": 0.95},
    )
    axes.set(
        ylabel=r"$B.subtilis\ \log(P)$",
        xlabel=r"$E.coli\ \log(P)$",
        xlim=lims,
        ylim=lims,
        xticks=xticks,
        yticks=xticks,
        yticklabels=xticklabels,
        aspect="equal",
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
    fig.savefig("figures/seqdeft_species_comparison.png", dpi=300)
    # fig.savefig("figures/seqdeft_species_comparison.svg", dpi=300)
