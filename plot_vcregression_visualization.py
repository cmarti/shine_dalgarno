import gpmap.src.plot.ds as dplot
import numpy as np
import pandas as pd
from gpmap.src.utils import read_edges
from plot_utils import (
    add_vcregression_labels,
    plot_path,
    plot_landscape,
    plot_relaxation_times,
)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    nodes_df = pd.read_parquet("results/vcregression.nodes.pq")
    relaxation_times = pd.read_csv("results/vcregression.decay_rates.csv")
    edges_df = read_edges("results/space.edges.npz")

    dsg = dplot.plot_edges(nodes_df, edges_df=edges_df, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((8.5, 8))
    axes = fig.axes[0]

    plot_landscape(
        axes,
        nodes_df,
        x="1",
        y="2",
        vmax=3.5,
        vmin=0.,
        cmap_label="log(GFP)",
        lims = (None, None),
        log_p=False,
    )
    axes.set(aspect="equal")

    times_axes = axes.inset_axes((0.8, 0.8, 0.225, 0.175))
    plot_relaxation_times(relaxation_times, times_axes)
    add_vcregression_labels(axes, nodes_df)
    plot_path(axes, nodes_df)

    fig.tight_layout()
    fig.savefig("figures/vcregression.visualization.png")
