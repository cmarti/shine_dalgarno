import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    species = 'b_sub'
    pred = pd.read_csv(
        "results/{}.seqdeft_path_contrasts.csv".format(species), index_col=0
    )
    print(pred)

    idx = pd.read_csv("data/peaks_contrasts.csv", index_col=0).index

    peaks_contrasts = pd.read_csv(
        "results/{}seqdeft_peaks_contrasts.csv".format(species),
        index_col=0,
    ).loc[idx, :]
    print(peaks_contrasts)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH * 0.525, FIG_WIDTH * 0.25),
        width_ratios=[2, 1],
    )

    axes = subplots[0]
    df = peaks_contrasts.loc[
        peaks_contrasts["background"] == "UUAAGGAGC", :
    ]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] - 0.125,
        -df["estimate"],
        yerr=2 * df["std"],
        fmt="o",
        lw=0,
        elinewidth=1,
        markersize=3.5,
        capsize=2.5,
        color="grey",
        label=r"UUA$\bf{AGGAG}$C",
    )

    df = peaks_contrasts.loc[
        peaks_contrasts["background"] == "UAAGGAGCA", :
    ]
    df["x"] = np.arange(df.shape[0])
    axes.errorbar(
        df["x"] + 0.125,
        -df["estimate"],
        yerr=2 * df["std"],
        fmt="o",
        markersize=3.5,
        lw=0,
        elinewidth=1,
        capsize=2.5,
        color="black",
        label=r"UA$\bf{AGGAG}$CA",
    )

    axes.set(
        xlabel="Mutation",
        ylabel="Scaled selection coefficient",
        xticks=df["x"],
        yticks=[-5, -2.5, 0, 2.5, 5],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=(0.035, 1.025), ncol=2)

    # Path posterior
    pred["step"] = np.arange(1, pred.shape[0] + 1)
    print(pred)
    axes = subplots[1]
    axes.errorbar(
        pred["step"],
        -pred["estimate"],
        yerr=2 * pred["std"],
        fmt="o",
        lw=0,
        markersize=3.5,
        elinewidth=1,
        capsize=2.5,
        color="black",
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
    fig.tight_layout(w_pad=1.5)
    fig.savefig(
        "figures/{}.seqdeft_path_posterior.svg".format(species), dpi=300
    )
