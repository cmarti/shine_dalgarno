import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, norm, beta
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    r2 = pd.read_csv("results/r2.csv", index_col=0)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0)
    vc_pred = pd.read_csv("results/vcregression.test.csv", index_col=0)
    mei_pred = pd.read_csv("results/mei.test.csv", index_col=0)
    test = test.join(vc_pred, rsuffix="_vc").join(mei_pred, rsuffix="_mei")
    test["pred_var_vc"] = test["y_var"] + test["y_var_vc"]
    test["pred_var_sd"] = np.sqrt(test["pred_var_vc"])

    calibration = []
    distrib = norm(loc=test["y_vc"], scale=np.sqrt(test["y_var_vc"]))
    for p in np.linspace(0.05, 0.95, 25):
        lower, upper = distrib.interval(p)
        inside = (test["y"] >= lower) & (test["y"] <= upper)
        a, b = 0.5 + inside.sum(), 0.5 + np.sum(~inside)
        lower, upper = beta(a, b).interval(0.95)
        obs_p = np.mean(inside)
        calibration.append(
            {
                "expected": p,
                "observed": obs_p,
                "lower": obs_p - lower,
                "upper": upper - obs_p,
            }
        )
    calibration = pd.DataFrame(calibration)

    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.27),
    )

    palette = {"VC": "black", "MEI": "grey"}
    axes = subplots[0]
    sns.lineplot(
        x="p",
        y="r2",
        hue="model",
        data=r2,
        ax=axes,
        palette=palette,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
        lw=0.75,
    )
    axes.set(
        xlabel="Fraction of training data",
        ylabel=r"Test $R^2$",
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
    )
    axes.legend(loc=4)
    axes.text(
        -0.3, 1.05, "A", fontsize=13, weight="bold", transform=axes.transAxes
    )

    axes = subplots[1]
    sns.lineplot(
        x="p",
        y="rmse",
        hue="model",
        data=r2,
        ax=axes,
        palette=palette,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
        lw=0.75,
    )
    axes.set(
        xlabel="Fraction of training data",
        ylabel=r"Test RMSE",
        xlim=(-0.05, 1.05),
        # ylim=(-0.05, 1.05),
    )
    axes.legend(loc=1)
    axes.text(
        -0.35, 1.05, "B", fontsize=13, weight="bold", transform=axes.transAxes
    )

    axes = subplots[2]
    axes.axline((0, 0), (1, 1), c="grey", linestyle="--", lw=0.5)
    axes.errorbar(
        calibration["expected"],
        calibration["observed"],
        yerr=calibration[["lower", "upper"]].values.T,
        c="black",
        lw=0,
        markersize=1.5,
        marker="o",
        capsize=0.75,
        elinewidth=0.75,
        capthick=0.75,
    )
    axes.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Expected fraction of test data\nwithin predictive interval",
        ylabel="Observed fraction of test data\nwithin predictive interval",
    )
    axes.text(
        -0.45, 1.05, "C", fontsize=13, weight="bold", transform=axes.transAxes
    )

    fig.tight_layout(w_pad=0)
    fig.savefig("figures/mave_predictions.png", dpi=300)
    fig.savefig("figures/mave_predictions.svg", dpi=300)
