import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    fig, subplots = plt.subplots(
        1,
        4,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.27),
        width_ratios=[2, 1, 2, 1],
    )

    # Load contrast data from SeqDEFT
    species = "e_coli"
    pred = pd.read_csv(
        "results/{}.seqdeft_path_contrasts.csv".format(species), index_col=0
    )
    idx = pd.read_csv("data/peaks_contrasts.csv", index_col=0).index
    peaks_contrasts = pd.read_csv(
        "results/{}.seqdeft_peaks_contrasts.csv".format(species),
        index_col=0,
    ).loc[idx, :]

    # Plot seqdeft mut effects
    axes = subplots[0]
    kwargs = {"fmt": "o", "lw": 0, "elinewidth": 1, "capsize": 2, "markersize": 3.5}
    df = peaks_contrasts.loc[peaks_contrasts["background"] == "UUAAGGAGC", :]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] - 0.125,
        -df["estimate"],
        yerr=2 * df["std"],
        color="grey",
        label=r"UUA$\bf{AGGAG}$C",
        **kwargs,
    )

    df = peaks_contrasts.loc[peaks_contrasts["background"] == "UAAGGAGCA", :]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] + 0.125,
        -df["estimate"],
        yerr=2 * df["std"],
        color="black",
        label=r"UA$\bf{AGGAG}$CA",
        **kwargs,
    )
    axes.set(
        xlabel="Mutation",
        ylabel="Scaled selection coefficient",
        xticks=df["x"],
        yticks=[-5, -2.5, 0, 2.5, 5],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=(-0.02, 1.025), ncol=2)
    axes.text(-0.3, 1.2, "D", fontsize=14, weight="bold", transform=axes.transAxes)

    # PLot phenotypic estimates along the +3 shift path
    pred["step"] = np.arange(1, pred.shape[0] + 1)
    axes = subplots[1]
    axes.errorbar(
        pred["step"],
        -pred["estimate"],
        yerr=2 * pred["std"],
        color="black",
        **kwargs,
    )

    axes.set(
        xlabel="Genetic background",
        xticks=pred["step"],
        ylabel="Scaled selection coefficient\n relative to UAAGGAGCA",
        # ylim=(0, 4.0),
        xlim=(0.5, 4.5),
    )
    labels = [
        r"$\bf{AGGAG}$AUAA",
        r"$\bf{AGGAGGAG}$A",
        r"UUA$\bf{AG}$AUAA",
        r"UUA$\bf{AGGAG}$A",
    ]
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.text(-0.6, 1.2, "E", fontsize=14, weight="bold", transform=axes.transAxes)

    # Load contrasts from VC regression
    pred = pd.read_csv("results/vcregression_path_contrasts.csv", index_col=0)
    peaks_contrasts = pd.read_csv(
        "results/vcregression_peaks_contrasts.csv",
        index_col=0,
    ).loc[idx, :]
    
    # Plot VCregression mut effects
    axes = subplots[2]
    df = peaks_contrasts.loc[peaks_contrasts["background"] == "UUAAGGAGC", :]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] - 0.125,
        df["estimate"],
        yerr=2 * df["std"],
        color="grey",
        label=r"UUA$\bf{AGGAG}$C",
        **kwargs,
    )

    df = peaks_contrasts.loc[peaks_contrasts["background"] == "UAAGGAGCA", :]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] + 0.125,
        df["estimate"],
        yerr=2 * df["std"],
        color="black",
        label=r"UA$\bf{AGGAG}$CA",
        **kwargs,
    )
    axes.set(
        xlabel="Mutation",
        ylabel="$\Delta$log(GFP)",
        xticks=df["x"],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=(-0.02, 1.025), ncol=2)
    axes.text(-0.25, 1.2, "F", fontsize=14, weight="bold", transform=axes.transAxes)

    # PLot phenotypic estimates along the +3 shift path
    pred["step"] = np.arange(1, pred.shape[0] + 1)
    axes = subplots[3]
    axes.errorbar(
        pred["step"],
        pred["estimate"],
        yerr=2 * pred["std"],
        color="black",
        **kwargs,
    )

    axes.set(
        xlabel="Genetic background",
        ylabel="log(GFP)",
        xticks=pred["step"],
        xlim=(0.5, 4.5),
    )
    labels = [
        r"$\bf{AGGAG}$AUAA",
        r"$\bf{AGGAGGAG}$A",
        r"UUA$\bf{AG}$AUAA",
        r"UUA$\bf{AGGAG}$A",
    ]
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.text(-0.5, 1.2, "G", fontsize=14, weight="bold", transform=axes.transAxes)

    fig.tight_layout(w_pad=0.5)
    fig.subplots_adjust(top=0.8)
    fig.savefig("figures/mut_eff_posterior.svg", dpi=300)
