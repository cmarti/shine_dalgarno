import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    print("Loading data from RNA model")
    energies = pd.read_csv("results/rna_model.pred.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0).join(energies)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0).join(energies)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH * 0.6, FIG_WIDTH * 0.25),
        sharex=True,
        sharey=True,
    )

    print("Plotting predicted vs observed phenotypes in training set")
    axes = subplots[0]
    axes.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=0.5)
    sns.histplot(
        x=train["y_pred"],
        y=train["y"],
        cmap="inferno",
        ax=axes,
        bins=100,
        cbar=True,
        cbar_kws={"label": "Number of sequences"},
        rasterized=True,
    )
    axes.set(
        xlabel="Training predicted log(GFP)",
        ylabel="Training measured log(GFP)",
        aspect="equal",
    )
    r2 = np.corrcoef(train["y_pred"], train["y"])[0, 1] ** 2
    axes.text(
        0.95,
        0.05,
        f"R$^2$={r2:.2f}",
        transform=axes.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
    )
    axes.text(
        -0.35, 1.05, "A", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plotting predicted vs observed phenotypes in test set")
    axes = subplots[1]
    axes.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=0.5)
    axes.scatter(x=test["y_pred"], y=test["y"], s=5, c="black", alpha=0.3, lw=0)
    axes.set(
        xlabel="Test predicted log(GFP)",
        ylabel="Test measured log(GFP)",
        xlim=(0, 3.5),
        ylim=(0, 3.5),
        aspect="equal",
    )
    r2 = np.corrcoef(test["y_pred"], test["y"])[0, 1] ** 2
    axes.text(
        0.95,
        0.05,
        f"R$^2$={r2:.2f}",
        transform=axes.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
    )
    axes.text(
        -0.35, 1.05, "B", fontsize=14, weight="bold", transform=axes.transAxes
    )

    fig.tight_layout(w_pad=0)
    fig.savefig("figures/rna_model_pred.png", dpi=300)
    fig.savefig("figures/rna_model_pred.svg", dpi=600)
