import gpmap.src.plot.ds as dplot
import numpy as np
import pandas as pd
from gpmap.src.utils import read_edges
from plot_utils import (
    annotate_seq,
    plot_path,
    plot_landscape,
    plot_relaxation_times,
    FIG_WIDTH,
)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    seqdeft_nodes_df = pd.read_parquet("results/seqdeft.nodes.pq")
    
    # print(seqdeft_nodes_df.loc[[x.startswith('GGAGG') for x in seqdeft_nodes_df.index], :].sort_values('function').tail(20))
    # exit()
    
    relaxation_times = pd.read_csv("results/seqdeft.decay_rates.csv")
    edges_df = read_edges("results/seqdeft.edges.npz")

    dsg = dplot.plot_edges(seqdeft_nodes_df, edges_df=edges_df, resolution=1200)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.66, FIG_WIDTH * 0.66))
    axes = fig.axes[0]

    plot_landscape(
        axes,
        seqdeft_nodes_df,
        x="1",
        y="2",
        vmax=np.log(5e-3),
        vmin=np.log(2e-8),
        cmap_label="log(P)",
        label_size=8,
    )
    axes.set(aspect="equal")

    times_axes = axes.inset_axes((0.825, 0.8, 0.225, 0.175))
    plot_relaxation_times(relaxation_times, times_axes)
    plot_path(axes, seqdeft_nodes_df, size=30)
    
    seqs = [
        "UAAGGAGCA",
        "UGAGGAGCA",
        "GGAGGAGCA",
        "GGAGGAGUA",
        "GGAGGAAUA",
    ]
    plot_path(axes, seqdeft_nodes_df, size=30, seqs=seqs)

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
    
    

    fig.tight_layout()
    fig.savefig("figures/seqdeft_visualization.png", dpi=300)
