import gpmap.src.plot.ds as dplot
import gpmap.src.plot.mpl as mplot
import numpy as np
import pandas as pd
from gpmap.src.utils import read_edges
from plot_utils import (
    annotate_seq,
    plot_path,
    plot_landscape,
    arrange_axis,
    plot_function_hist,
    FIG_WIDTH,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    nodes_df = pd.read_parquet("results/thermodynamic_model.nodes.pq")
    edges_df = read_edges("results/seqdeft.edges.npz")

    dsg = dplot.plot_edges(nodes_df, edges_df=edges_df, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.45, FIG_WIDTH * 0.45))
    axes = fig.axes[0]

    nodes_hist_axes = axes.inset_axes((0.0, 0.88, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.0, 0.85, 0.3, 0.02))

    vmin, vmax = 0, 3.5
    mplot.plot_nodes(
        axes,
        nodes_df,
        sort_by="function",
        sort_ascending=True,
        size=3,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label="log(GFP)",
        cbar_orientation="horizontal",
    )

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_cbar_axes.set_xticklabels(
        nodes_cbar_axes.get_xticklabels(), fontsize=6
    )
    nodes_cbar_axes.set_xlabel("log(GFP)", fontsize=7)
    ticks = [-2.0, -1, 0, 1, 2, 3, 4]
    # lims = [-3-1, 3.]
    arrange_axis(axes, "1", "2", ticks, None, fontsize=8, xpos=0.5, ypos=0.45)
    axes.set(
        xticks=ticks,
        yticks=ticks,
        ylim=(-3, 3.5),
        xlim=(-2.5, 3.5),
        aspect="equal",
    )

    seqs = [
        "AGGAGGUAC",
        "AGGAGGAAC",
        "AGGAGGAGC",
        "AGGAGGAGG",
        "UGGAGGAGG",
        "UUGAGGAGG",
        "UUAAGGAGG",
    ]
    plot_path(axes, nodes_df, size=20, seqs=seqs)

    seqs = [
        "CAGGAGGUA",
        "GAGGAGGUA",
        "GAGGAGGUU",
        "GAGGAGGAU",
        "GAGGAGAAU",
        "GAGGAUAAU",
        "GAGGUUAAU",
        "GAGUUUAAU",
    ]
    plot_path(axes, nodes_df, size=20, seqs=seqs)

    seqs = [
        "UAAGGAGGU",
        "UGAGGAGGU",
        "GGAGGAGGU",
        "GGAGGAGGC",
        "GGAGGAGCC",
        "GGAGGAACC",
        "GGAGGUACC",
    ]
    plot_path(axes, nodes_df, size=20, seqs=seqs)

    kwargs = {"fontsize": 7, "arrow_size": 0.35}
    annotate_seq(
        axes,
        "CAGGAGGUA",
        nodes_df,
        dx=0.5,
        dy=0.05,
        ha="left",
        va="center",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UUAAGGAGG",
        nodes_df,
        dx=1.2,
        dy=0.2,
        ha="left",
        va="center",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAAGGAGGU",
        nodes_df,
        dx=0.0,
        dy=1.1,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGAGG",
        nodes_df,
        dx=-0.3,
        dy=-0.3,
        ha="right",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGUAC",
        nodes_df,
        dx=0.9,
        dy=0.4,
        ha="center",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GAGUUUAAU",
        nodes_df,
        dx=1.2,
        dy=0.3,
        ha="left",
        va="center",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GAGGAGGAU",
        nodes_df,
        dx=0.5,
        dy=0,
        ha="left",
        va="center",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GGAGGUACC",
        nodes_df,
        dx=0.2,
        dy=-0.3,
        ha="center",
        va="top",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GGAGGAGGU",
        nodes_df,
        dx=-0.3,
        dy=0.5,
        ha="center",
        va="bottom",
        **kwargs,
    )

    fig.tight_layout()
    fig.savefig("figures/thermodynamic_model_visualization.png", dpi=300)
