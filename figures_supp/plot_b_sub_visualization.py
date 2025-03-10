import gpmap.src.plot.ds as dplot
import numpy as np
import pandas as pd
from gpmap.src.utils import read_edges
from plot_utils import (
    annotate_seq,
    plot_path,
    plot_landscape,
    FIG_WIDTH,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    seqdeft_nodes_df = pd.read_parquet("results/b_sub.seqdeft.nodes.pq")
    print(seqdeft_nodes_df.sort_values("1"))
    print(seqdeft_nodes_df.sort_values("2"))

    relaxation_times = pd.read_csv("results/b_sub.seqdeft.decay_rates.csv")
    edges_df = read_edges("results/b_sub.seqdeft.edges.npz")

    dsg = dplot.plot_edges(seqdeft_nodes_df, edges_df=edges_df, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.6, FIG_WIDTH * 0.6))
    axes = fig.axes[0]

    cbar_ax, hist_ax = plot_landscape(
        axes,
        seqdeft_nodes_df,
        # edf=edges_df,
        x="1",
        y="2",
        size=3,
        vmax=np.log(5e-3),
        vmin=np.log(1e-9),
        lims=(-2.75, 2.75),
        cmap_label="log(P)",
        xpos=0.425,
        label_size=8,
    )
    yticklabels = 10 ** np.arange(-8, -2).astype(float)
    cbar_ax.set_xlabel("Sequence probability", fontsize=8)
    hist_ax.set_ylabel("Frequency", fontsize=8)
    cbar_ax.set_xticks(np.log(yticklabels))
    cbar_ax.set_xticklabels(
        [
            "$10^{-8}$",
            "$10^{-7}$",
            "$10^{-6}$",
            "$10^{-5}$",
            "$10^{-4}$",
            "$10^{-3}$",
        ],
        fontsize=6,
        rotation=45,
    )
    axes.set(
        aspect="equal",
        yticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5],
    )

    kwargs = {"fontsize": 7, "arrow_size": 0.25}
    # Annotate corner sequences
    annotate_seq(
        axes,
        "UUAGGAGGA",
        seqdeft_nodes_df,
        dx=-0.2,
        dy=-0.2,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AAGGAGCUG",
        seqdeft_nodes_df,
        dx=0.1,
        dy=-0.3,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "CAAAGGAGG",
        seqdeft_nodes_df,
        dx=0,
        dy=0.4,
        ha="center",
        va="top",
        **kwargs,
    )

    # Plot paths and annotations
    seqs = [
        "CAAAGGAGG",
        "CAGAGGAGG",
        "CGGAGGAGG",
        "AGGAGGAGG",
        "AGGAGGAGA",
        "AGGAGGAUA",
        "AGGAGGUUA",
    ]
    plot_path(axes, seqdeft_nodes_df, lw=1, size=20, seqs=seqs)

    seqs = [
        "UUAGGAGGA",
        "UGAGGAGGA",
        "GGAGGAGGA",
        "GGAGGAGUA",
        "GGAGGAUUA",
        "GGAGGUUUA",
    ]
    plot_path(axes, seqdeft_nodes_df, lw=1, size=20, seqs=seqs)

    annotate_seq(
        axes,
        "AGGAGGAGG",
        seqdeft_nodes_df,
        dx=0.2,
        dy=0.5,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGUUA",
        seqdeft_nodes_df,
        dx=0.3,
        dy=0.2,
        ha="left",
        va="top",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GGAGGAGGA",
        seqdeft_nodes_df,
        dx=-0.3,
        dy=0.2,
        ha="right",
        va="bottom",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GGAGGUUUA",
        seqdeft_nodes_df,
        dx=-0.5,
        dy=0.3,
        ha="right",
        va="bottom",
        **kwargs,
    )

    axes.set_ylim((-2, 3.0))
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.savefig("figures/b_sub.seqdeft_visualization.png", dpi=300)
