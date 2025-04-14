import pandas as pd
import matplotlib.pyplot as plt

from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mut_effs_posterior,
    plot_path_posterior,
)


if __name__ == "__main__":
    fig, subplots = plt.subplots(
        1,
        4,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.28),
        width_ratios=[2, 1, 2, 1],
    )
    panely = 1.2

    print("Load contrast results")
    fpath = "results/vcregression.contrasts.csv"
    vc_contrasts = pd.read_csv(fpath, index_col=0)

    fpath = "results/e_coli.seqdeft.contrasts.csv"
    seqdeft_contrasts = pd.read_csv(fpath, index_col=0)

    print("Plot SeqDEFT mutational effects")
    axes = subplots[0]
    plot_mut_effs_posterior(axes, seqdeft_contrasts)
    axes.set(ylabel="Scaled selection coefficient")
    axes.text(
        -0.3, panely, "D", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot SeqDEFT estimates along the +3 shift path")
    axes = subplots[1]
    plot_path_posterior(axes, seqdeft_contrasts)
    axes.set(
        ylabel="Scaled selection coefficient\n relative to sequence average",
    )
    axes.text(
        -0.6, panely, "E", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot VC regression mutational effects")
    axes = subplots[2]
    plot_mut_effs_posterior(axes, vc_contrasts)
    axes.text(
        -0.25, panely, "F", fontsize=14, weight="bold", transform=axes.transAxes
    )

    print("Plot VC regression estimates along the +3 shift path")
    axes = subplots[3]
    plot_path_posterior(axes, vc_contrasts)
    axes.text(
        -0.5, panely, "G", fontsize=14, weight="bold", transform=axes.transAxes
    )

    fig.tight_layout(w_pad=0.5)
    fig.subplots_adjust(top=0.8)
    fig.savefig("figures/mut_eff_posterior.png", dpi=300)
    fig.savefig("figures/mut_eff_posterior.svg", dpi=300)
