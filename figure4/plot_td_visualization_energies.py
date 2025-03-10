import gpmap.src.plot.ds as dplot
import gpmap.src.plot.mpl as mplot
import holoviews as hv
import pandas as pd
import seaborn as sns
from gpmap.src.utils import read_edges

from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    hv.extension("matplotlib")

    energies = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    nodes_df = pd.read_parquet("results/thermodynamic_model.nodes.pq").join(
        energies
    )
    edges_df = read_edges("results/seqdeft.edges.npz")

    x, y = "1", "2"
    edges_dsg = dplot.plot_edges(
        nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    ).opts(padding=0.1)
    grid = hv.Layout(edges_dsg + edges_dsg + edges_dsg)
    grid.opts(sublabel_format="")
    fig = dplot.dsg_to_fig(grid)
    fig.subplots_adjust(wspace=0.1)
    fig.set_size_inches((FIG_WIDTH * 0.625, FIG_WIDTH * 0.225))

    for i, axes in enumerate(fig.axes):
        nodes_cbar_axes = axes.inset_axes((-0.1, 0.7, 0.03, 0.35))
        mplot.plot_nodes(
            axes,
            nodes_df,
            x=x,
            y=y,
            sort_by="dg{}".format(i + 3),
            sort_ascending=False,
            color="dg{}".format(i + 3),
            size=2,
            cmap="Greys_r",
            cbar_axes=nodes_cbar_axes,
            cbar_orientation="vertical",
            vmin=-2,
            vmax=10,
        )
        nodes_cbar_axes.set_ylabel(
            "$\Delta G$ at {} (kcal/mol)".format(-14 + i), fontsize=6
        )
        yticks = [-2, 2, 6, 10]
        nodes_cbar_axes.set_yticks(yticks)
        nodes_cbar_axes.set_yticklabels(yticks, fontsize=6)
        axes.set(aspect="equal", xlabel="", ylabel="")
        axes.spines["left"].set(position=("data", 0), zorder=0, alpha=0.5)
        axes.spines["bottom"].set(position=("data", 0), zorder=0, alpha=0.5)
        axes.plot(
            (1),
            (0),
            ls="",
            marker=">",
            ms=2,
            color="k",
            transform=axes.get_yaxis_transform(),
            clip_on=False,
        )
        axes.plot(
            (0),
            (1),
            ls="",
            marker="^",
            ms=2,
            color="k",
            transform=axes.get_xaxis_transform(),
            clip_on=False,
        )
        axes.annotate(
            "Diffusion\naxis {}".format(y),
            xy=(1.0, 0.98),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            va="center",
        )
        axes.annotate(
            "Diffusion\naxis {}".format(x),
            xy=(0.975, 0.315),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            va="bottom",
        )
        sns.despine(ax=axes)

    fig.savefig(
        "figures/thermodynamic_model_visualization.energies.png", dpi=300
    )
