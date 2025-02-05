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
    plot_landscape,
    plot_relaxation_times,
    arrange_axis
)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    mfs = [1, 1.5, 2, 2.5]
    lims = (-4.5, 5.)
    ticks = [-4, -3, -2, 1, 0, 1, 2, 3, 4]
    nplots = len(mfs)
    edges_df = read_edges("results/space.edges.npz")
    nodes_df = {mf: pd.read_parquet("results/vcregression.map.mf_{}.nodes.pq".format(mf))
                for mf in mfs}

    # print('Plotting edges')
    # dsg = dplot.plot_edges(nodes_df[mfs[0]], edges_df=edges_df, resolution=800)
    # for mf in mfs[1:]:
    #     dsg += dplot.plot_edges(nodes_df[mf], edges_df=edges_df, resolution=800)
    
    # fig = dplot.dsg_to_fig(dsg)
    # fig.set_size_inches((FIG_WIDTH, FIG_WIDTH / nplots))
    # subplots = fig.axes
    fig, subplots = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / nplots))
    
    print('Plotting nodes')
    for mf, axes in zip(mfs, subplots):
        df = nodes_df[mf]
        if df.loc['AGGAGAAUA', '3'] < 0:
            df['3'] = -df['3']
        print(mf)
        mplot.plot_nodes(axes, df, cbar_label='log(GFP)', vmin=0, vmax=3.5,
                         cbar=False, sort_by='3', sort_ascending=True, size=1.5)
        axes.set(aspect="equal",
                 xlim=lims, ylim=lims)
        add_vcregression_labels(axes, df, label_path=False, arrow_size=0.2)
        plot_path(axes, df, size=10, lw=1)
        arrange_axis(axes, x='1', y='2', ticks=ticks, lims=lims,
                     fontsize=7, xpos=0.5, ypos=0.55, ms=3)

    fig.tight_layout(w_pad=0.1)
    fig.savefig("figures/vcregression.visualization.mfs.png", dpi=300)
