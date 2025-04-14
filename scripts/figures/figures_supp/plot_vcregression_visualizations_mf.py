import matplotlib
import gpmap.plot.mpl as mplot
import pandas as pd
import matplotlib.pyplot as plt
from gpmap.utils import read_edges
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    add_vcregression_labels,
    plot_path,
    annotate_seq,
    arrange_axis,
)


if __name__ == "__main__":
    matplotlib.use("Agg")

    mfs = [1, 1.5, 2, 2.5]
    lims = (-4.5, 5.0)
    ticks = [-4, -3, -2, 1, 0, 1, 2, 3, 4]
    nplots = len(mfs)

    print("Load input data")
    edges_df = read_edges("results/edges.npz")
    fname = "results/vcregression.map.mf_{}.nodes.pq"
    nodes_df = {mf: pd.read_parquet(fname.format(mf)) for mf in mfs}

    fig, subplots = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / nplots))
    cbar_ax = subplots[1].inset_axes((-0, 0.7, 0.03, 0.3))
    vmin, vmax = 0, 3.5
    print("Plotting nodes")
    for mf, axes in zip(mfs, subplots):
        print("\tMean function at stationarity: {:.2f}".format(mf))
        df = nodes_df[mf]
        if df.loc["AGGAGAAUA", "3"] < 0:
            df["3"] = -df["3"]
        mplot.plot_edges(
            axes, df, edges_df=edges_df, alpha=0.02, rasterized=True
        )
        mplot.plot_nodes(
            axes,
            df,
            cbar_label="log(GFP)",
            cbar=True,
            cbar_axes=cbar_ax,
            cbar_orientation="vertical",
            sort_by="3",
            sort_ascending=True,
            size=1.5,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        if mf in [1.5, 2.0]:
            add_vcregression_labels(axes, df, label_path=False, arrow_size=0.2)
        plot_path(axes, df, size=10, lw=1, vmin=vmin, vmax=vmax)
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
        plot_path(axes, df, size=10, lw=1, seqs=seqs, vmin=vmin, vmax=vmax)

        seqs = [
            "UAAGGAGCA",
            "UGAGGAGCA",
            "GGAGGAGCA",
            "GGAGGAGAA",
            "GGAGGAUAA",
            "GGAGGUUAA",
            "GGAGUUUAA",
        ]
        plot_path(axes, df, size=10, lw=1, seqs=seqs, vmin=vmin, vmax=vmax)
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
    fig.savefig("figures/vcregression_visualization_mfs.png", dpi=300)
    fig.savefig("figures/vcregression_visualization_mfs.svg", dpi=300)
