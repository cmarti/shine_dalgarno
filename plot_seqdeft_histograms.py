import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    inferred = pd.read_csv("results/seqdeft_inference.csv", index_col=0)
    inferred["logQ"] = np.log10(inferred["Q_star"])
    max_counts = int(inferred["counts"].max())

    fig, subplots = plt.subplots(
        max_counts + 1, 1, figsize=(FIG_WIDTH * 0.25, FIG_WIDTH * 0.25), sharex=True
    )

    bins = np.linspace(inferred["logQ"].min(), inferred["logQ"].max(), 50)

    for axes, (c, df) in zip(subplots, inferred.groupby("counts")):

        sns.histplot(
            df["logQ"],
            bins=bins,
            ax=axes,
            stat="percent",
            color="grey",
            alpha=0.8,
            lw=0,
            label="Estimated probability",
        )
        sns.despine(ax=axes, bottom=False, left=True)
        ylim = axes.get_ylim()
        axes.axvline(
            np.log10(df["frequency"][0]),
            color="black",
            lw=0.75,
            linestyle="-",
            label="Observed frequency",
        )
        axes.set(ylabel="", yticklabels=[], yticks=[])
        label = "N$_i$={} (n={})".format(int(np.round(c, 0)), df.shape[0])
        if c == 0:
            label = "N$_i$=0"
        axes.text(
            0.01,
            0.95,
            label,
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )

    sns.despine(ax=axes, bottom=False)
    xticks = np.arange(-8, -2)
    xticklabels = [
        "$10^{-8}$",
        "$10^{-7}$",
        "$10^{-6}$",
        "$10^{-5}$",
        "$10^{-4}$",
        "$10^{-3}$",
    ]
    axes.set(
        xticks=xticks, xticklabels=xticklabels, xlabel="Sequence probability"
    )

    fig.supylabel("# sequences", fontsize=7, x=0.05, y=0.525)

    fig.subplots_adjust(bottom=0.2, right=0.95)
    fig.savefig("figures/seqdeft_probs.svg")
    fig.savefig("figures/seqdeft_probs.png", dpi=300)
