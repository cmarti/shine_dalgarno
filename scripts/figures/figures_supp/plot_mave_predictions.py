import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, beta
from scripts.figures.plot_utils import FIG_WIDTH


def calc_model_calibration(y, means, vars):
    calibration = []
    distrib = norm(loc=means, scale=np.sqrt(vars))
    for p in np.linspace(0.05, 0.95, 25):
        lower, upper = distrib.interval(p)
        inside = (y >= lower) & (y <= upper)
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
    return calibration


def plot_model_calibration(axes, calibration, color, label):
    axes.axline((0, 0), (1, 1), c="grey", linestyle="--", lw=0.5)
    axes.errorbar(
        calibration["expected"],
        calibration["observed"],
        yerr=calibration[["lower", "upper"]].values.T,
        c=color,
        label=label,
        lw=0,
        markersize=1.5,
        marker="o",
        capsize=0.75,
        elinewidth=0.75,
        capthick=0.75,
        alpha=0.5,
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


if __name__ == "__main__":
    print("Loading input data")
    r2 = pd.read_csv("results/models.r2.csv", index_col=0)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0)
    vc_pred = pd.read_csv("results/vcregression.test_pred.csv", index_col=0)
    mei_pred = pd.read_csv("results/mei.test_pred.csv", index_col=0)

    print("Preparing data for plotting")
    test = test.join(vc_pred, rsuffix="_vc").join(mei_pred, rsuffix="_mei")
    y, f, f_var = test["y"], test["f"], test["f_var"]
    vc_calibration = calc_model_calibration(y, f, f_var)
    y, f, f_var = test["y"], test["f_mei"], test["f_var_mei"]
    mei_calibration = calc_model_calibration(y, f, f_var)

    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.27),
    )

    print("Plotting CV curves for VC regression and MEI: R2")
    palette = {"VC": "black", "MEI": "grey"}
    axes = subplots[0]
    sns.lineplot(
        x="p",
        y="r2",
        hue="model",
        hue_order=["MEI", "VC"],
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

    print("Plotting CV curves for VC regression and MEI: RMSE")
    axes = subplots[1]
    sns.lineplot(
        x="p",
        y="rmse",
        hue="model",
        hue_order=["MEI", "VC"],
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

    print("Plotting calibration curves for VC regression and MEI")
    axes = subplots[2]
    plot_model_calibration(axes, mei_calibration, color="grey", label="MEI")
    plot_model_calibration(axes, vc_calibration, color="black", label="VC")
    axes.legend(loc=4)

    fig.tight_layout(w_pad=0)
    fig.savefig("figures/mave_predictions.png", dpi=300)
    fig.savefig("figures/mave_predictions.svg", dpi=300)
