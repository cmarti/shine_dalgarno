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

    print("Loading VC regression visualization data")
    nodes_df = pd.read_parquet("results/vcregression.map.mf_2.nodes.pq")
    edges_df = read_edges("results/seqdeft.edges.npz")

    fig, subplots = plt.subplots(
        1, 4, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.035]}
    )
    nodes_cbar_axes = subplots[-1]

    diffusion_axes = [
        ("1", "2", "3"),
        ("1", "3", "2"),
        ("3", "4", "1"),
    ]

    print("Generating views of different axes")
    for axes, (x, y, z) in zip(subplots[:3], diffusion_axes):
        print("\tDiffusion axes {} and {}".format(x, y))
        mplot.plot_edges(
            axes, nodes_df, edges_df=edges_df, x=x, y=y, alpha=0.02
        )
        mplot.plot_nodes(
            axes,
            nodes_df,
            x=x,
            y=y,
            sort_by=z,
            sort_ascending=True,
            color="function",
            size=5,
            cmap="viridis",
            cbar_axes=nodes_cbar_axes,
            cbar_orientation="vertical",
        )
        nodes_cbar_axes.set_ylabel("log(GFP)")

        axes.set(
            aspect="equal",
            xlabel="",
            ylabel="",
            # ylim=(-3.1, 3.1), xlim=(-3.1, 3.1),
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
        axes.tick_params(labelsize=8)
        sns.despine(ax=axes)
        axes.set(aspect="equal")
        plot_path(axes, nodes_df, x=x, y=y, size=20)
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
        plot_path(axes, nodes_df, x=x, y=y, size=20, seqs=seqs)

        seqs = [
            "UAAGGAGCA",
            "UGAGGAGCA",
            "GGAGGAGCA",
            "GGAGGAGAA",
            "GGAGGAUAA",
            "GGAGGUUAA",
            "GGAGUUUAA",
        ]
        plot_path(axes, nodes_df, x=x, y=y, size=20, seqs=seqs)

    fontsize = 8
    axes = subplots[0]
    x, y, z = diffusion_axes[0]
    axes.text(
        1,
        0.5,
        "Diffusion\naxis {}".format(x),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )
    axes.text(
        0.45,
        1,
        "Diffusion\naxis {}".format(y),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )

    kwargs = {"x": x, "y": y, "fontsize": fontsize}
    annotate_seq(
        axes,
        "GAGUUUAAU",
        nodes_df,
        dx=-0.8,
        dy=0.2,
        ha="right",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAGGAGGUA",
        nodes_df,
        dx=-0.8,
        dy=-0.35,
        ha="right",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        nodes_df,
        dx=0,
        dy=-0.75,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        nodes_df,
        dx=0.2,
        dy=-0.6,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        nodes_df,
        dx=1.1,
        dy=0.9,
        ha="left",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        nodes_df,
        dx=0.7,
        dy=0.2,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GAGGAGGAU",
        nodes_df,
        dx=-0.7,
        dy=0.2,
        ha="right",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GGAGUUUAA",
        nodes_df,
        dx=-0.4,
        dy=0.4,
        ha="right",
        va="bottom",
        **kwargs,
    )

    axes = subplots[1]
    x, y, z = diffusion_axes[1]
    axes.text(
        1,
        0.38,
        "Diffusion\naxis {}".format(x),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )
    axes.text(
        0.45,
        1,
        "Diffusion\naxis {}".format(y),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )

    kwargs = {"x": x, "y": y, "fontsize": fontsize}
    annotate_seq(
        axes,
        "GAGUUUAAU",
        nodes_df,
        dx=-0.6,
        dy=0,
        ha="right",
        va="center",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAGGAGGUA",
        nodes_df,
        dx=0.6,
        dy=-0.35,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        nodes_df,
        dx=0,
        dy=-0.75,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        nodes_df,
        dx=0.2,
        dy=-0.6,
        ha="center",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        nodes_df,
        dx=0.9,
        dy=0.7,
        ha="left",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGGAGC",
        nodes_df,
        dx=0.6,
        dy=0.7,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GAGGAGGAU",
        nodes_df,
        dx=1.0,
        dy=-1.4,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GGAGUUUAA",
        nodes_df,
        dx=-0.4,
        dy=0.4,
        ha="right",
        va="bottom",
        **kwargs,
    )

    axes = subplots[2]
    x, y, z = diffusion_axes[2]
    axes.text(
        1,
        0.42,
        "Diffusion\naxis {}".format(x),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )
    axes.text(
        0.6,
        1,
        "Diffusion\naxis {}".format(y),
        transform=axes.transAxes,
        fontsize=9,
        ha="center",
        va="top",
    )

    kwargs = {"x": x, "y": y, "fontsize": fontsize}
    annotate_seq(
        axes,
        "GAGUUUAAU",
        nodes_df,
        dx=0,
        dy=0.8,
        ha="center",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        nodes_df,
        dx=-0.2,
        dy=-0.6,
        ha="right",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        nodes_df,
        dx=-0.2,
        dy=-1.0,
        ha="right",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GGAGGUACA",
        nodes_df,
        dx=0.5,
        dy=0,
        ha="left",
        va="center",
        **kwargs,
    )
    annotate_seq(
        axes,
        "AGGAGAAUA",
        nodes_df,
        dx=0.9,
        dy=-0.9,
        ha="left",
        va="top",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GAGGAGGAU",
        nodes_df,
        dx=0.3,
        dy=1.3,
        ha="left",
        va="bottom",
        **kwargs,
    )
    annotate_seq(
        axes,
        "GGAGUUUAA",
        nodes_df,
        dx=0.4,
        dy=-0.1,
        ha="left",
        va="top",
        **kwargs,
    )

    print("Saving figure")
    fig.tight_layout(w_pad=0.2)
    nodes_cbar_axes.set_position([0.925, 0.2, 0.0075, 0.6])
    fig.savefig("figures/vcregression.visualization.axis6.png", dpi=300)
    print("Done")
