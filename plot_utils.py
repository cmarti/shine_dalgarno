import gpmap.src.plot.mpl as plot
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.patches as patches
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm

FIG_WIDTH = 8

# Fonts
plt.rcParams["font.family"] = "Nimbus Sans"
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["legend.labelspacing"] = 0.1

# plt.rcParams["axes.titlepad"] = 3

# Linewidths
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.width"] = 0.35
plt.rcParams["ytick.minor.width"] = 0.35



def annotate_seq(
    axes, seq, label, df, dx, dy, ha, va, x="1", y="2", fontsize=None,
    arrow_size=1,
):
    x, y = df.loc[seq, [x, y]]
    axes.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=1 * arrow_size,
            headwidth=6 * arrow_size,
            headlength=10* arrow_size,
        ),
        ha=ha,
        va=va,
        fontsize=fontsize,
    )


def plot_path(axes, nodes, x="1", y="2", size=40, lw=1.5):
    seqs = [
        "AGGAGAAUA",
        "AGGAGGAUA",
        "AGGAGGAGA",
        "AGGAGGAGC",
        "UGGAGGAGC",
        "UUGAGGAGC",
        "UUAAGGAGC",
    ]
    sl = len(seqs)
    edf = pd.DataFrame({"i": np.arange(sl - 1), "j": np.arange(1, sl)})
    ndf = nodes.loc[seqs, :]
    v = nodes["function"]
    vmax = v.max()
    plot.plot_nodes(
        axes,
        ndf,
        lw=lw,
        size=size,
        cbar=False,
        vmin=-15,
        vmax=vmax,
        color="function",
        zorder=4,
        x=x,
        y=y,
    )
    plot.plot_edges(
        axes, ndf, edf, color="black", alpha=1, width=lw, zorder=3, x=x, y=y
    )


def plot_landscape(
    axes,
    ndf,
    edf=None,
    x="1",
    y="2",
    cmap_label="",
    vmin=np.log(1e-7),
    vmax=np.log(1e-3),
    c="function",
    z="3",
    lims = (-3.1, 3.1),
    log_p=True,
):
    nodes_hist_axes = axes.inset_axes((0.05, 0.875, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.05, 0.85, 0.3, 0.02))

    plot.plot_nodes(
        axes,
        ndf,
        x=x,
        y=y,
        sort_by=z,
        sort_ascending=True,
        color=c,
        size=5,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label=cmap_label,
        cbar_orientation="horizontal",
        vmin=vmin,
        vmax=vmax,
    )

    plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c)
    if log_p:
        arrange_cbar(nodes_cbar_axes)
        ticks = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    else:
        ticks = np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    arrange_axis(axes, x, y, ticks, lims)


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c):
    bins = np.linspace(vmin, vmax, 30)
    plot.plot_color_hist(nodes_hist_axes, ndf[c], cmap="viridis", bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=10)


def arrange_cbar(nodes_cbar_axes):
    yticklabels = 10 ** np.arange(-7, -2).astype(float)
    nodes_cbar_axes.set_xlabel("Sequence probability", fontsize=10)
    nodes_cbar_axes.set_xticks(np.log(yticklabels))
    nodes_cbar_axes.set_xticklabels(
        ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"]
    )


def arrange_axis(axes, x, y, ticks, lims, fontsize=12,
                 xpos=0.52, ypos=0.52, ms=5):
    axes.set(aspect="equal", xlabel="", ylabel="")
    axes.spines["left"].set(position=("data", 0), zorder=0, alpha=0.5)
    axes.spines["bottom"].set(position=("data", 0), zorder=0, alpha=0.5)
    axes.set(xticks=ticks, yticks=ticks, ylim=lims, xlim=lims)
    axes.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=ms,
        color="k",
        transform=axes.get_yaxis_transform(),
        clip_on=False,
    )
    axes.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=ms,
        color="k",
        transform=axes.get_xaxis_transform(),
        clip_on=False,
    )
    axes.text(
        1.1,
        xpos,
        "Diffusion axis {}".format(x),
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )
    axes.text(
        ypos,
        1,
        "Diffusion axis {}".format(y),
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
    )
    sns.despine(ax=axes)


def plot_relaxation_times(relaxation_times, axes):
    axes.scatter(
        relaxation_times["k"],
        relaxation_times["relaxation_time"],
        c="black",
        s=10,
    )
    axes.plot(
        relaxation_times["k"],
        relaxation_times["relaxation_time"],
        c="black",
        lw=1,
    )
    axes.axhline(0.25, c="grey", lw=0.75, label="Neutral", linestyle="--")
    axes.legend(loc=1)
    axes.set(
        ylim=(0.2, None),
        # xticks=relaxation_times["k"],
        xticks=[1, 5, 10, 15, 20],
    )
    axes.set_ylabel("Relaxation time\n(expected substitutions)", fontsize=10)
    axes.set_xlabel("Diffusion axis", fontsize=10)
    
    
def add_vcregression_labels(axes, nodes_df, fontsize=6, label_path=True, arrow_size=1):
    annotate_seq(
        axes,
        "AAGGAGCAG",
        r"A$\bf{AGGAG}$CAG",
        nodes_df,
        dx=0.7,
        dy=0.7,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        r"UA$\bf{AGGAG}$CA",
        nodes_df,
        dx=-0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        r"UUA$\bf{AGGAG}$C",
        nodes_df,
        dx=0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    if label_path:
        annotate_seq(
            axes,
            "AGGAGAAUA",
            r"$\bf{AGGAG}$AAUA",
            nodes_df,
            dx=0.2,
            dy=-1.2,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            arrow_size=arrow_size,
        )
        annotate_seq(
            axes,
            "AGGAGGAGC",
            r"$\bf{AGGAGGAG}$A",
            nodes_df,
            dx=0.1,
            dy=-0.8,
            ha="left",
            va="top",
            fontsize=fontsize,
            arrow_size=arrow_size,
        )
