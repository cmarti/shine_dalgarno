import gpmap.src.plot.ds as dplot
import numpy as np
import pandas as pd
from gpmap.src.utils import read_edges
from plot_utils import (
    annotate_seq,
    plot_path,
    plot_landscape,
    plot_relaxation_times,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    seqdeft_nodes_df = pd.read_parquet("data/seqdeft.nodes.pq")
    relaxation_times = pd.read_csv("data/seqdeft.decay_rates.csv")
    edges_df = read_edges("data/seqdeft.edges.npz")

    dsg = dplot.plot_edges(seqdeft_nodes_df, edges_df=edges_df, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((8.5, 8))
    axes = fig.axes[0]

    plot_landscape(
        axes,
        seqdeft_nodes_df,  # edf=edges_df,
        x="1",
        y="2",
        vmax=np.log(5e-3),
        vmin=np.log(2e-8),
        cmap_label="log(P)",
    )
    axes.set(aspect="equal")

    times_axes = axes.inset_axes((0.8, 0.8, 0.225, 0.175))
    plot_relaxation_times(relaxation_times, times_axes)

    annotate_seq(
        axes,
        "AAGGAGCAG",
        r"A$\bf{AGGAG}$CAG",
        seqdeft_nodes_df,
        dx=0.3,
        dy=0.3,
        ha="left",
        va="bottom",
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        r"UUA$\bf{AGGAG}$C",
        seqdeft_nodes_df,
        dx=0,
        dy=-0.5,
        ha="center",
        va="top",
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        r"UA$\bf{AGGAG}$CA",
        seqdeft_nodes_df,
        dx=0.2,
        dy=-0.5,
        ha="left",
        va="top",
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        r"$\bf{AGGAG}$AAUA",
        seqdeft_nodes_df,
        dx=0.7,
        dy=0.5,
        ha="left",
        va="bottom",
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        r"$\bf{AGGAGGAG}$A",
        seqdeft_nodes_df,
        dx=0.6,
        dy=-0.35,
        ha="left",
        va="top",
    )

    plot_path(axes, seqdeft_nodes_df)

    fig.tight_layout()
    fig.savefig("figures/seqdeft_visualization.png")
