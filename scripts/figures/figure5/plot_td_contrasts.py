import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils import get_contrast_matrix
from scripts.figures.plot_utils import (
    FIG_WIDTH,
)


if __name__ == "__main__":
    fig, subplots = plt.subplots(
        1,
        4,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.28),
        width_ratios=[2, 1, 2, 1],
    )
    panely = 1.2

    print("Load thermodynamic model inferred energies")
    energies = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    min_energy = energies.iloc[:, 1:].values.min()
    energies.iloc[:, 1:] -= min_energy

    contrast_matrix = get_contrast_matrix()
    contrasts = (
        contrast_matrix.values.T @ energies.loc[contrast_matrix.index, :].values
    )
    contrasts = pd.DataFrame(
        contrasts, index=contrast_matrix.columns, columns=energies.columns
    )

    print("Plot model mutational effects")
    axes = subplots[0]
    peaks_contrasts = ["in" in x for x in contrasts.index]
    peaks_contrasts = contrasts.loc[peaks_contrasts, :].copy()
    items = [x.split("_") for x in peaks_contrasts.index.values]
    peaks_contrasts["mutation"] = [x[0] for x in items]
    peaks_contrasts["background"] = [x[-1] for x in items]

    bc1 = peaks_contrasts["background"] == "UUAAGGAGC"
    df = peaks_contrasts.loc[bc1, :].copy()
    df["x"] = np.arange(df.shape[0]) - 0.125
    axes.scatter(
        df["x"],
        df["y_pred"],
        s=10,
        c="grey",
        label=r"UUA$\bf{AGGAG}$C",
    )
    bc2 = peaks_contrasts["background"] == "UAAGGAGCA"
    df = peaks_contrasts.loc[bc2, :].copy()
    df["x"] = np.arange(df.shape[0]) + 0.125
    axes.scatter(
        df["x"],
        df["y_pred"],
        s=10,
        color="black",
        label=r"UA$\bf{AGGAG}$CA",
    )
    axes.set(
        xlabel="Mutation",
        ylabel="$\Delta$log(GFP)",
        ylim=(-1.5, 1.5),
        xticks=df["x"],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=(-0.02, 1.025), ncol=2)

    axes.text(
        -0.3, panely, "E", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot model predictions along the +3 shift path")
    axes = subplots[1]
    seqs = ["AGGAGGNNN", "NGGAGGAGN", "NNNAGGNNN", "NNNAGGAGG"]
    labels = [
        r"$\bf{AGGAGG}$NNN",
        r"N$\bf{GGAGGAG}$N",
        r"NNN$\bf{AGG}$NNN",
        r"NNN$\bf{AGGAGG}$",
    ]
    df = contrasts.loc[seqs, :].copy()
    df["step"] = np.arange(1, df.shape[0] + 1)
    axes.scatter(
        df["step"],
        df["y_pred"],
        s=10,
        color="black",
    )
    if "NNNNNNNNN" in contrasts.index:
        axes.axhline(
            contrasts.loc["NNNNNNNNN", "y_pred"],
            linestyle="--",
            c="grey",
            lw=0.75,
            label="Average",
        )
    if "AAGGAGGUG" in contrasts.index:
        axes.axhline(
            contrasts.loc["AAGGAGGUG", "y_pred"],
            linestyle="--",
            c="black",
            lw=0.75,
            label="AAGGAGGUG",
        )
    axes.set(
        xlabel="Genetic background",
        ylabel="log(GFP)",
        xticks=df["step"],
        xlim=(0.5, 4.5),
        ylim=(None, 2.9),
    )

    axes.set_xticklabels(labels, rotation=45, ha="right")

    axes.text(
        -0.6, panely, "F", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot mutational effects on register-specific energies")
    axes = subplots[2]
    bc1 = peaks_contrasts["background"] == "UUAAGGAGC"
    df = peaks_contrasts.loc[bc1, :].copy()
    df["x"] = np.arange(df.shape[0]) - 0.125
    axes.scatter(
        df["x"],
        -df["dg6"],
        s=10,
        c="black",
        label="Position -11",
    )
    bc2 = peaks_contrasts["background"] == "UAAGGAGCA"
    df = peaks_contrasts.loc[bc2, :].copy()
    df["x"] = np.arange(df.shape[0]) + 0.125
    axes.scatter(
        df["x"],
        -df["dg7"],
        s=10,
        color="grey",
        label="Position -10",
    )
    axes.set(
        xlabel="Mutation",
        ylabel="$-\Delta\Delta G$ (kcal/mol)",
        ymargin=0.2,
        xticks=df["x"],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=3)

    axes.text(
        -0.25, panely, "G", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot register energies along the +3 shift path")
    axes = subplots[3]

    seqs = ["AGGAGGNNN", "NGGAGGAGN", "NNNAGGNNN", "NNNAGGAGG"]
    labels = [
        r"$\bf{AGGAGG}$NNN",
        r"N$\bf{GGAGGAG}$N",
        r"NNN$\bf{AGG}$NNN",
        r"NNN$\bf{AGGAGG}$",
    ]
    df = contrasts.loc[seqs, :].copy()
    df["step"] = np.arange(1, df.shape[0] + 1)
    axes.scatter(
        df["step"] - 0.15,
        -df["dg4"],
        s=10,
        color="black",
        label="Position -13",
    )
    axes.scatter(
        df["step"] + 0.15,
        -df["dg7"],
        s=10,
        color="grey",
        label="Position -10",
    )
    axes.legend(loc=(0.1, 1.05))
    if "NNNNNNNNN" in contrasts.index:
        axes.axhline(
            -contrasts.loc["NNNNNNNNN", "dg6"],
            linestyle="--",
            c="grey",
            lw=0.75,
            label="Average",
        )
    axes.axhline(
        0,
        linestyle="--",
        c="black",
        lw=0.75,
        label="AAGGAGGUG",
    )
    axes.set(
        xlabel="Genetic background",
        ylabel="$-\Delta G$ (kcal/mol)",
        xticks=df["step"],
        xlim=(0.5, 4.5),
        ymargin=0.2,
    )

    axes.set_xticklabels(labels, rotation=45, ha="right")

    axes.text(
        -0.5, panely, "H", fontsize=14, weight="bold", transform=axes.transAxes
    )

    fig.tight_layout(w_pad=0.5)
    fig.subplots_adjust(top=0.8)
    fig.savefig("figures/mut_eff_td_model.png", dpi=300)
    fig.savefig("figures/mut_eff_td_model.svg", dpi=300)
