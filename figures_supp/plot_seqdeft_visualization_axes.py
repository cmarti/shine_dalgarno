import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as mplot

from gpmap.src.utils import read_edges
from plot_utils import (
    annotate_seq,
    plot_path,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    print("Loading SeqDEFT visualization data")
    seqdeft_nodes_df = pd.read_parquet("results/seqdeft.nodes.pq")
    edges_df = read_edges("results/seqdeft.edges.npz")

    fig, subplots = plt.subplots(
        1, 4, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.035]}
    )
    nodes_cbar_axes = subplots[-1]

    diffusion_axes = [
        ("1", "2", "3"),
        ("1", "3", "2"),
        ("2", "3", "1"),
    ]

    print("Generating views of different axes")
    for axes, (x, y, z) in zip(subplots[:3], diffusion_axes):
        print("\tDiffusion axes {} and {}".format(x, y))
        mplot.plot_edges(
            axes, seqdeft_nodes_df, edges_df=edges_df, x=x, y=y, alpha=0.02
        )
        mplot.plot_nodes(
            axes,
            seqdeft_nodes_df,
            x=x,
            y=y,
            sort_by=z,
            sort_ascending=True,
            color="function",
            size=5,
            cmap="viridis",
            cbar_axes=nodes_cbar_axes,
            cbar_orientation="vertical",
            vmax=np.log(5e-3),
            vmin=np.log(2e-8),
        )
        yticklabels = 10 ** np.arange(-7, -2).astype(float)
        nodes_cbar_axes.set_ylabel("Sequence probability", fontsize=9)
        nodes_cbar_axes.set_yticks(np.log(yticklabels))
        nodes_cbar_axes.set_yticklabels(
            ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"]
        )

        axes.set(
            aspect="equal",
            xlabel="",
            ylabel="",
            ylim=(-2.8, 3.5),
            xlim=(-3.2, 3.2),
        )
        axes.spines["left"].set(position=("data", 0), zorder=0, alpha=0.5)
        axes.spines["bottom"].set(position=("data", 0), zorder=0, alpha=0.5)
        axes.plot(
            (1),
            (0),
            ls="",
            marker=">",
            ms=5,
            color="k",
            transform=axes.get_yaxis_transform(),
            clip_on=False,
        )
        axes.plot(
            (0),
            (1),
            ls="",
            marker="^",
            ms=5,
            color="k",
            transform=axes.get_xaxis_transform(),
            clip_on=False,
        )
        axes.text(
            1,
            0.35,
            "Diffusion\naxis {}".format(x),
            transform=axes.transAxes,
            fontsize=9,
            ha="center",
            va="top",
        )
        axes.text(
            0.35,
            1,
            "Diffusion\naxis {}".format(y),
            transform=axes.transAxes,
            fontsize=9,
            ha="center",
            va="top",
        )
        axes.tick_params(labelsize=8)
        sns.despine(ax=axes)

        axes.set(aspect="equal")
        plot_path(axes, seqdeft_nodes_df, x=x, y=y, size=30)
        seqs = [
            "UAAGGAGCA",
            "UGAGGAGCA",
            "GGAGGAGCA",
            "GGAGGAGUA",
            "GGAGGAAUA",
        ]
        plot_path(axes, seqdeft_nodes_df, x=x, y=y, size=30, seqs=seqs)

    fontsize = 8
    axes = subplots[0]
    x, y, z = ("1", "2", "3")
    axes.set(aspect="equal")
    annotate_seq(
        axes,
        "AAGGAGCAG",
        seqdeft_nodes_df,
        dx=0.6,
        dy=-0.35,
        ha="left",
        va="top",
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        seqdeft_nodes_df,
        dx=0,
        dy=-0.5,
        ha="center",
        va="top",
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        seqdeft_nodes_df,
        dx=0.2,
        dy=-0.5,
        ha="left",
        va="top",
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        seqdeft_nodes_df,
        dx=0.7,
        dy=0.5,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        seqdeft_nodes_df,
        dx=0.6,
        dy=0.7,
        ha="left",
        va="top",
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "GGAGGAAUA",
        seqdeft_nodes_df,
        dx=-0.4,
        dy=0.9,
        ha="right",
        va="bottom",
        fontsize=fontsize,
    )

    axes = subplots[1]
    x, y, z = ("1", "3", "2")
    annotate_seq(
        axes,
        "AAGGAGCAG",
        seqdeft_nodes_df,
        dx=0.2,
        dy=-0.9,
        ha="left",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        seqdeft_nodes_df,
        dx=0,
        dy=-0.5,
        ha="center",
        va="top",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        seqdeft_nodes_df,
        dx=0.2,
        dy=-0.5,
        ha="left",
        va="top",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        seqdeft_nodes_df,
        dx=0.7,
        dy=0.5,
        ha="left",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        seqdeft_nodes_df,
        dx=0.6,
        dy=0.35,
        ha="left",
        va="top",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "GGAGGAAUA",
        seqdeft_nodes_df,
        dx=-0.6,
        dy=0.7,
        ha="right",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )

    axes = subplots[2]
    x, y, z = ("2", "3", "1")
    annotate_seq(
        axes,
        "AAGGAGCAG",
        seqdeft_nodes_df,
        dx=0.2,
        dy=-0.9,
        ha="center",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        seqdeft_nodes_df,
        dx=0,
        dy=-0.5,
        ha="center",
        va="top",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        seqdeft_nodes_df,
        dx=-0.3,
        dy=0.2,
        ha="right",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        seqdeft_nodes_df,
        dx=0.7,
        dy=0.3,
        ha="left",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        seqdeft_nodes_df,
        dx=-0.6,
        dy=0.45,
        ha="right",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )
    annotate_seq(
        axes,
        "GGAGGAAUA",
        seqdeft_nodes_df,
        dx=-0.6,
        dy=0.3,
        ha="right",
        va="bottom",
        x=x,
        y=y,
        fontsize=fontsize,
    )

    print("Saving figure")
    fig.tight_layout()
    nodes_cbar_axes.set_position([0.9, 0.2, 0.0075, 0.6])
    fig.savefig("figures/seqdeft.visualization.axis3.png", dpi=300)
    print("Done")
