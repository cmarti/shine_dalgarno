import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, norm, beta
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    loss = pd.read_csv("results/thermodynamic_model.ll.csv", index_col=0)
    td_pred = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0).join(td_pred)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0).join(td_pred)

    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.27),
        width_ratios=[0.8, 1, 0.8],
    )

    palette = {"VC": "black", "MEI": "grey"}

    axes = subplots[0]
    axes.plot(loss["ll"], c="black", lw=0.75)
    axes.set(
        xlabel="Iteration number",
        ylabel="log-likelihood",
    )
    axes.axhline(loss["ll"].max(), linestyle="--", color="grey", lw=0.5)
    axes.text(
        -0.35, 1.05, "A", fontsize=14, weight="bold", transform=axes.transAxes
    )

    axes = subplots[1]
    axes.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=0.5)
    sns.histplot(
        x=train["y_pred"],
        y=train["y"],
        cmap="inferno",
        ax=axes,
        bins=100,
        cbar=True,
        cbar_kws={"label": "Number of sequences"},
    )
    axes.set(
        xlabel="Training predicted log(GFP)",
        ylabel="Training measured log(GFP)",
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
        -0.35, 1.05, "B", fontsize=14, weight="bold", transform=axes.transAxes
    )

    axes = subplots[2]
    axes.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=0.5)
    axes.scatter(x=test["y_pred"], y=test["y"], s=5, c="black", alpha=0.3, lw=0)
    axes.set(
        xlabel="Test predicted log(GFP)",
        ylabel="Test measured log(GFP)",
        xlim=(0.1, 3),
        ylim=(0.1, 3),
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
        -0.35, 1.05, "C", fontsize=14, weight="bold", transform=axes.transAxes
    )

    fig.tight_layout(w_pad=0)
    fig.savefig("figures/td_fit.png", dpi=300)
    fig.savefig("figures/td_fit.svg", dpi=300)
