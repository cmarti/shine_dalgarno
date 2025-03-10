import gpmap.src.plot.ds as dplot
import gpmap.src.plot.mpl as mplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpmap.src.utils import read_edges
from plot_utils import (
    FIG_WIDTH,
    add_vcregression_labels,
    plot_path,
    annotate_seq,
    arrange_axis,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    mfs = [1, 1.5, 2, 2.5]
    lims = (-4.5, 5.0)
    ticks = [-4, -3, -2, 1, 0, 1, 2, 3, 4]
    nplots = len(mfs)
    edges_df = read_edges("results/seqdeft.edges.npz")
    nodes_df = {
        mf: pd.read_parquet(
            "results/vcregression.map.mf_{}.nodes.pq".format(mf)
        )
        for mf in mfs
    }

    # print('Plotting edges')
    # dsg = dplot.plot_edges(nodes_df[mfs[0]], edges_df=edges_df, resolution=800)
    # for mf in mfs[1:]:
    #     dsg += dplot.plot_edges(nodes_df[mf], edges_df=edges_df, resolution=800)

    # fig = dplot.dsg_to_fig(dsg)
    # fig.set_size_inches((FIG_WIDTH, FIG_WIDTH / nplots))
    # subplots = fig.axes
    fig, subplots = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / nplots))
    cbar_ax = subplots[1].inset_axes((-0, 0.7, 0.03, 0.3))

    print("Plotting nodes")
    for mf, axes in zip(mfs, subplots):
        df = nodes_df[mf]
        if df.loc["AGGAGAAUA", "3"] < 0:
            df["3"] = -df["3"]
        mplot.plot_edges(axes, df, edges_df=edges_df, alpha=0.02)
        mplot.plot_nodes(
            axes,
            df,
            cbar_label="log(GFP)",
            vmin=0,
            vmax=3.5,
            cbar=True,
            cbar_axes=cbar_ax,
            cbar_orientation="vertical",
            sort_by="3",
            sort_ascending=True,
            size=1.5,
        )
        if mf in [1.5, 2.0]:
            add_vcregression_labels(axes, df, label_path=False, arrow_size=0.2)
        plot_path(axes, df, size=10, lw=1)
        seqs = [
            "UAGGAGGUA",
            "GAGGAGGUA",
            "GAGGAGGUU",
            "GAGGAGGAU",
            "GAGGAUGAU",
            "GAGGUUGAU",
            "GAGUUUGAU",
            "GAGUUUAAU",
        ]
        plot_path(axes, df, size=10, lw=1, seqs=seqs)

        seqs = [
            "UAAGGAGCA",
            "UGAGGAGCA",
            "GGAGGAGCA",
            "GGAGGAGAA",
            "GGAGGAUAA",
            "GGAGGUUAA",
            "GGAGUUUAA",
        ]
        plot_path(axes, df, size=10, lw=1, seqs=seqs)
        arrange_axis(
            axes,
            x="1",
            y="2",
            ticks=ticks,
            lims=lims,
            fontsize=7,
            xpos=0.5,
            ypos=0.45,
            ms=3,
        )
        axes.set(aspect="equal", xlim=(-3.5, 5), ylim=(-4, 4.5))
        axes.set_title("Average log(GFP)={:.2f}".format(mf), fontsize=8)
    cbar_ax.set_ylabel("log(GFP)", fontsize=6)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=6)

    axes = subplots[-1]
    annotate_seq(
        axes,
        "UAGGAGGUA",
        nodes_df[2.5],
        dx=0.7,
        dy=-0.2,
        ha="left",
        va="top",
        fontsize=6,
        arrow_size=0.2,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        nodes_df[2.5],
        dx=-0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=6,
        arrow_size=0.2,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        nodes_df[2.5],
        dx=0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=6,
        arrow_size=0.2,
    )

    fig.tight_layout(w_pad=0.1)
    fig.savefig("figures/vcregression.visualization.mfs.png", dpi=300)
