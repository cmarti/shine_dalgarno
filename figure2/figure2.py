import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from scipy.stats import pearsonr
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    upstream_bases = 20
    distance_start_codon = 4
    seq_length = 9
    pos_labels = [str(x) for x in range(-13, -4)]
    sd_start = upstream_bases - distance_start_codon - seq_length
    x1, x2 = 0.015, 0.5
    y1, y2, y3 = 0.965, 0.64, 0.34

    print("Loading data")
    dcor = pd.read_csv(
        "results/dmsc.empirical_distance_correlation.csv", index_col=0
    )
    vc = pd.read_csv("results/vc.prior_variance_components.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0)

    full = pd.read_csv("results/vcregression.full.csv", index_col=0)
    pred = pd.read_csv("results/vcregression.test.csv", index_col=0)
    test = test.join(pred, rsuffix="_pred")
    train = train.join(full, rsuffix="_pred")

    marginal_sites = pd.read_csv(
        "results/vcregression.map_site_marginal_epistasis.csv", index_col=0
    )
    marginal_pw = pd.read_csv(
        "results/vcregression.map_pairwise_marginal_epistasis.csv", index_col=0
    )

    fig = plt.figure(figsize=(FIG_WIDTH * 0.5, 0.575 * FIG_WIDTH))
    gs = gs.GridSpec(3, 4, width_ratios=[1, 0.05, 1, 0.05])

    print("Plotting empirical distance correlation")
    axes = fig.add_subplot(gs[0, 0])
    axes.scatter(dcor["d"], dcor["rho"], color="black", s=7.5)
    axes.plot(dcor["d"], dcor["rho"], color="black", lw=1)
    axes.set(
        ylabel="Empirical correlation",
        ylim=(None, 1.05),
        xlabel="Hamming distance",
        xticks=np.arange(dcor.shape[0]),
        yticks=np.linspace(0, 1, 6),
    )
    axes.axhline(0, linestyle="--", color="grey", lw=0.75)
    axes.text(
        x1, y1, "A", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting prior variance components")
    axes = fig.add_subplot(gs[0, 2])
    axes.bar(x=vc["k"], height=vc["variance_perc"], color="black")
    axes.set(
        xlabel="Interaction order $k$",
        ylabel="% variance explained",
        xticks=np.arange(1, 10),
        ylim=(0, 52.5),
    )

    axes = axes.twinx()
    axes.scatter(vc["k"], np.cumsum(vc["variance_perc"]), color="grey", s=7.5)
    axes.plot(vc["k"], np.cumsum(vc["variance_perc"]), color="grey", lw=1)
    axes.set(ylabel="% cumulative variance", ylim=(0, 105))
    axes.text(
        x2, y1, "B", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting predicted vs. observed in training set")
    axes = fig.add_subplot(gs[1, 0])
    cbar_axes1 = fig.add_subplot(gs[1, 1])
    lims = (full["y"].min() - 0.5, full["y"].max() + 0.5)
    bins = np.linspace(lims[0], lims[-1], 100)
    x, y = train["y_pred"], train["y"]
    r2 = pearsonr(x, y)[0] ** 2
    sns.histplot(
        x=x,
        y=y,
        cmap="inferno",
        ax=axes,
        bins=(bins, bins),
        cbar_ax=cbar_axes1,
        cbar=True,
        cbar_kws={"label": "# sequences ($10^{3}$)", "shrink": 0.95},
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        xticks=np.arange(4),
        xlabel="Predicted log(GFP)",
        ylabel="Measured log(GFP)",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
    axes.text(
        0.95,
        0.05,
        "Training data",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axes.text(
        0.05,
        0.95,
        "$R^2$" + "={:.2f}".format(r2),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )
    axes.text(
        x1, y2, "C", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting predicted vs. observed in test set")
    axes = fig.add_subplot(gs[1, 2])
    predictive_std = np.sqrt(test["y_var_pred"])
    x, xerr = test["y_pred"], 2 * predictive_std
    y, yerr = test["y"], 2 * np.sqrt(test["y_var"])
    r2 = pearsonr(y, x)[0] ** 2
    lower = test["y_pred"] - 2 * predictive_std
    upper = test["y_pred"] + 2 * predictive_std

    axes.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color="black",
        fmt="o",
        alpha=0.2,
        markeredgewidth=0,
        markersize=2.5,
        lw=0.5,
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        xticks=np.arange(4),
        xlabel="Predicted log(GFP)",
        ylabel="Measured log(GFP)",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
    axes.text(
        0.05,
        0.95,
        "$R^2$"
        + "={:.2f}\nn={}".format(
            r2, test.shape[0]
        ),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )
    axes.text(
        0.95,
        0.05,
        "Test data",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axes.text(
        x2, y2, "D", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting site marginal variance components in the MAP")
    axes = fig.add_subplot(gs[2, 0])
    cbar_axes2 = fig.add_subplot(gs[2, 1])
    marginal_sites.index = pos_labels
    label = "% variance explained by\n interactions involving site $i$"
    sns.heatmap(
        marginal_sites.T.iloc[::-1, :] * 100,
        ax=axes,
        cmap="Greys",
        cbar_ax=cbar_axes2,
        cbar_kws={"label": label, "shrink": 0.8},
        vmin=0,
        vmax=20,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(
        ylabel="Order of interaction $k$", xlabel="Position $i$ relative to AUG"
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    axes.text(
        x1, y3, "E", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print(
        "Plotting marginal variance components between pairs of sites in the MAP"
    )
    axes = fig.add_subplot(gs[2, 2])
    cbar_axes3 = fig.add_subplot(gs[2, 3])
    label = "% variance explained by\n interactions involving $i,j$"
    marginal_pw['high_order'] = marginal_pw['sum'] - marginal_pw['2']
    df1 = (
        pd.pivot_table(
            marginal_pw, index="i", columns="j", values="sum"
        ).fillna(0)
        * 100
    )
    # df2 = (
    #     pd.pivot_table(
    #         marginal_pw, index="i", columns="j", values="high_order"
    #     ).fillna(0)
    #     * 100
    # )
    
    m = np.zeros((9, 9))
    m[:-1, 1:] = df1.values
    m[1:, :-1] += df1.values.T
    df = pd.DataFrame(m, index=pos_labels, columns=pos_labels)

    sns.heatmap(
        df,
        ax=axes,
        cmap="Greys",
        cbar_kws={"label": label, "shrink": 0.8},
        vmin=0,
        vmax=20,
        cbar_ax=cbar_axes3,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(
        xlabel="Position $i$ relative to AUG",
        ylabel="Position $j$ relative to AUG",
    )
    axes.text(
        x2, y3, "F", fontsize=10, weight="bold", transform=fig.transFigure
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    print("Adjusting panel positions")
    fig.subplots_adjust(
        bottom=0.1, left=0.125, hspace=0.5, wspace=1.0, top=0.95, right=0.975
    )
    for cbar_axes in [cbar_axes1, cbar_axes2, cbar_axes3]:
        pos = cbar_axes.get_position()
        pos.p0[0] -= 0.105
        pos.p1[0] -= 0.105
        cbar_axes.set_position(pos)
        sns.despine(ax=cbar_axes, top=False, right=False)
    cbar_axes1.set_yticklabels([0, 5, 10, 15, 20])
    # cbar_axes2.set_yticks([0, 5, 10, 15, 20])
    # fig.subplots_adjust(hspace=0.2)
    fig.savefig("figures/figure2.png", dpi=300)
