import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import holoviews as hv
import pandas as pd
import seaborn as sns
from gpmap.utils import read_edges

from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    hv.extension("matplotlib")

    energies = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    print(energies.corr())
    min_energy = energies.min().min()
    nodes_df = pd.read_parquet("results/thermodynamic_model.nodes.pq").join(
        energies
    )
    edges_df = read_edges("results/seqdeft.edges.npz")

    x, y = "1", "2"
    edges_dsg = dplot.plot_edges(
        nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    ).opts(padding=0.1)
    dsg = edges_dsg + edges_dsg + edges_dsg + edges_dsg + edges_dsg + edges_dsg
    grid = hv.Layout(dsg).cols(2).opts(sublabel_format="")
    fig = dplot.dsg_to_fig(grid)
    fig.set_size_inches((FIG_WIDTH * 0.3, FIG_WIDTH * 0.5))
    # fig.set_size_inches((FIG_WIDTH * 0.25 * 1.5, FIG_WIDTH * 0.45 * 1.5))

    nodes_cbar_axes = fig.axes[0].inset_axes((-0.135, 0.65, 0.035, 0.35))
    positions = [2, 5, 3, 6, 4, 1]

    for i, axes in zip(positions, fig.axes):
        col = "dg{}".format(i + 1)
        nodes_df[col] = nodes_df[col] - min_energy
        mplot.plot_nodes(
            axes,
            nodes_df,
            x=x,
            y=y,
            sort_by=col,
            # sort_by="3",
            sort_ascending=False,
            color=col,
            size=1,
            cmap="Greys_r",
            # cbar=False,
            cbar=True,
            cbar_axes=nodes_cbar_axes,
            cbar_orientation="vertical",
            vmin=0,
            vmax=10,
            rasterized=True,
        )
        nodes_cbar_axes.set_ylabel("$\Delta G$ (kcal/mol)", fontsize=6)
        yticks = [0, 2.5, 5, 7.5, 10]
        nodes_cbar_axes.set_yticks(yticks)
        nodes_cbar_axes.set_yticklabels(yticks, fontsize=6)
        axes.set(
            aspect="equal",
            xlabel="",
            ylabel="",
        )
        axes.set_title("Position {}".format(-16 + i), fontsize=6)
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
        # axes.annotate(
        #     "Diffusion axis {}".format(y),
        #     xy=(1.4, 0.98),
        #     xycoords=("data", "axes fraction"),
        #     textcoords="offset points",
        #     fontsize=6,
        #     ha="center",
        #     va="center",
        # )
        # axes.annotate(
        #     "Diffusion\naxis {}".format(x),
        #     xy=(1.0, 0.2),
        #     xycoords=("axes fraction", "data"),
        #     textcoords="offset points",
        #     fontsize=6,
        #     ha="center",
        #     va="bottom",
        # )
        sns.despine(ax=axes)

    # fig.subplots_adjust()
    fig.savefig(
        "figures/thermodynamic_model_visualization.energies.png", dpi=300
    )
    fig.savefig(
        "figures/thermodynamic_model_visualization.energies.svg", dpi=600
    )
