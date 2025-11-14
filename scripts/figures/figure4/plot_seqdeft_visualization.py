import gpmap.plot.ds as dplot
import numpy as np
import pandas as pd
import matplotlib

from gpmap.utils import read_edges
from scripts.figures.plot_utils import (
    annotate_seq,
    plot_path,
    plot_landscape,
    plot_relaxation_times,
    FIG_WIDTH,
)


if __name__ == "__main__":
    matplotlib.use("Agg")
    print("Loading input data")
    seqdeft_nodes_df = pd.read_parquet("results/e_coli.seqdeft.map.nodes.pq")
    relaxation_times = pd.read_csv("results/e_coli.seqdeft.map.decay_rates.csv")
    edges_df = read_edges("results/edges.npz")

    print("Plotting visualization")
    dsg = dplot.plot_edges(seqdeft_nodes_df, edges_df=edges_df, resolution=1200)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.66, FIG_WIDTH * 0.66))
    axes = fig.axes[0]
    vmin, vmax = np.log(2e-8), np.log(5e-3)
    plot_landscape(
        axes,
        seqdeft_nodes_df,
        x="1",
        y="2",
        vmax=vmax,
        vmin=vmin,
        cmap_label="log(P)",
        label_size=8,
    )
    axes.set(aspect="equal")

    print("Adding paths and sequence labels")
    times_axes = axes.inset_axes((0.825, 0.8, 0.225, 0.175))
    plot_relaxation_times(relaxation_times, times_axes)
    plot_path(axes, seqdeft_nodes_df, size=30, vmin=vmin, vmax=vmax)

    seqs = [
        "UAAGGAGCA",
        "UGAGGAGCA",
        "GGAGGAGCA",
        "GGAGGAGUA",
        "GGAGGAAUA",
    ]
    plot_path(axes, seqdeft_nodes_df, size=30, seqs=seqs, vmin=vmin, vmax=vmax)

    fontsize = 7
    annotate_seq(
        axes,
        "AAGGAGCAG",
        seqdeft_nodes_df,
        dx=0.1,
        dy=0.3,
        ha="left",
        va="bottom",
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
        dy=-0.35,
        ha="left",
        va="top",
        fontsize=fontsize,
    )

    annotate_seq(
        axes,
        "AAGGAAUAU",
        seqdeft_nodes_df,
        dx=-0.1,
        dy=0.6,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    annotate_seq(
        axes,
        "GGAGGAGAA",
        seqdeft_nodes_df,
        dx=-0.4,
        dy=0.1,
        ha="right",
        va="bottom",
        fontsize=fontsize,
    )

    annotate_seq(
        axes,
        "GGAGGAAUA",
        seqdeft_nodes_df,
        dx=0,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=fontsize,
    )

    print("Rendering plot")
    fig.tight_layout()
    fig.savefig("figures/seqdeft_visualization.png", dpi=300)
    fig.savefig("figures/seqdeft_visualization.svg", dpi=600)
