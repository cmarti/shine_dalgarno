import pandas as pd
import matplotlib.pyplot as plt

from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mut_effs_posterior,
    plot_path_posterior,
)

if __name__ == "__main__":
    print("Load contrast results")
    fpath = "results/b_sub.seqdeft.contrasts.csv"
    contrasts = pd.read_csv(fpath, index_col=0)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH * 0.525, FIG_WIDTH * 0.27),
        width_ratios=[2, 1],
    )

    print("Plot SeqDEFT mutational effects")
    axes = subplots[0]
    plot_mut_effs_posterior(axes, contrasts)
    axes.set(ylabel="Scaled selection coefficient")

    # Path posterior
    print("Plot SeqDEFT estimates along the +3 shift path")
    axes = subplots[1]
    plot_path_posterior(axes, contrasts)
    axes.set(
        ylabel="Scaled selection coefficient\n relative to sequence average",
    )

    fig.tight_layout(w_pad=1.5)
    fpath = "figures/b_sub.seqdeft.contrasts.svg"
    fig.savefig(fpath, dpi=300)
    fpath = "figures/b_sub.seqdeft.contrasts.png"
    fig.savefig(fpath, dpi=300)
