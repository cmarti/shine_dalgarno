import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import pandas as pd
from gpmap.utils import read_edges
from scripts.figures.plot_utils import (
    annotate_seq,
    plot_path,
    arrange_axis,
    plot_function_hist,
    FIG_WIDTH,
)


if __name__ == "__main__":
    x, y, z = '1', '2', '3'
    
    import matplotlib

    matplotlib.use("Agg")

    nodes_df = pd.read_parquet("results/rnamodel.nodes.pq")
    edges_df = read_edges("results/seqdeft.edges.npz")

    dsg = dplot.plot_edges(nodes_df, edges_df=edges_df, x=x, y=y, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.45, FIG_WIDTH * 0.45))
    axes = fig.axes[0]

    nodes_hist_axes = axes.inset_axes((0.0, 0.88, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.0, 0.85, 0.3, 0.02))

    vmin, vmax = 0, 3.5
    mplot.plot_nodes(
        axes,
        nodes_df,
        x=x,
        y=y,
        # sort_by="function",
        sort_ascending=True,
        sort_by=z,
        size=3,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label="log(GFP)",
        cbar_orientation="horizontal",
        rasterized=True,
    )

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_cbar_axes.set_xticklabels(
        nodes_cbar_axes.get_xticklabels(), fontsize=6
    )
    nodes_cbar_axes.set_xlabel("log(GFP)", fontsize=7)
    ticks = [-2.0, -1, 0, 1, 2, 3, 4]
    # lims = [-3-1, 3.]
    arrange_axis(axes, x, y, ticks, None, fontsize=8, xpos=0.365, ypos=0.4)
    axes.set(
        xticks=ticks,
        yticks=ticks,
        # ylim=(-4, 3),
        # xlim=(-3.5, 3.5),
        ylim=(-2.5, 4.75),
        xlim=(-2.5, 4.5),
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
    plot_path(axes, nodes_df, size=20, seqs=seqs, vmin=vmin, vmax=vmax, x=x, y=y)

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
    plot_path(axes, nodes_df, size=20, seqs=seqs, vmin=vmin, vmax=vmax, x=x, y=y)

    seqs = [
        "UAAGGAGGU",
        "UGAGGAGGU",
        "GGAGGAGGU",
        "GGAGGAGGC",
        "GGAGGAGCC",
        "GGAGGAACC",
        "GGAGGUACC",
    ]
    plot_path(axes, nodes_df, size=20, seqs=seqs, vmin=vmin, vmax=vmax, x=x, y=y)

    kwargs = {"fontsize": 7, "arrow_size": 0.35, 'x': x, 'y': y}
    annotate_seq(
        axes,
        "CAGGAGGUA",
        nodes_df,
        dx=-0.3,
        dy=0.3,
        ha="right",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GAGUUUAAU",
        nodes_df,
        dx=-0.1,
        dy=-0.6,
        ha="right",
        va="top",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GAGGAGGAU",
        nodes_df,
        dx=-0.2,
        dy=0.4,
        ha="right",
        va="bottom",
        **kwargs,
    )
    
    annotate_seq(
        axes,
        "UUAAGGAGG",
        nodes_df,
        dx=-0.5,
        dy=0.0,
        ha="right",
        va="center",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGAGG",
        nodes_df,
        dx=0.1,
        dy=0.75,
        ha="left",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGUAC",
        nodes_df,
        dx=0.5,
        dy=0.,
        ha="left",
        va="center",
        **kwargs,
    )
    
    annotate_seq(
        axes,
        "UAAGGAGGU",
        nodes_df,
        dx=0.2,
        dy=-1.3,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GGAGGUACC",
        nodes_df,
        dx=-0.6,
        dy=0.5,
        ha="right",
        va="bottom",
        **kwargs,
    )

    annotate_seq(
        axes,
        "GGAGGAGGU",
        nodes_df,
        dx=-0.8,
        dy=1.5,
        ha="right",
        va="bottom",
        **kwargs,
    )

    fig.tight_layout()
    fig.savefig("figures/rnamodel_visualization.png", dpi=300)
    fig.savefig("figures/rnamodel_visualization.svg", dpi=600)
