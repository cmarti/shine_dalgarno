import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gpmap.plot.mpl import plot_hyperparam_cv
from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    logL_df = pd.read_csv(
        "results/thermodynamic_model_delta.cv_results.csv", index_col=0
    )

    logL = logL_df.groupby("a")["ll"].mean()
    ainf = logL.loc[np.inf]
    a0 = logL.loc[0]
    print(logL)
    fig, axes = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * 0.25, FIG_WIDTH * 0.225)
    )

    sns.lineplot(
        x="a",
        y="ll",
        data=logL_df,
        ax=axes,
        errorbar="sd",
        color="black",
        err_style="bars",
        err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
        lw=0.75,
    )
    # axes.axhline(logL.max(), lw=0.5, linestyle="--", color="red")
    axes.axhline(ainf, lw=0.5, linestyle="--", color="red", alpha=0.5)
    axes.axhline(a0, lw=0.5, linestyle="--", color="red", alpha=0.5)
    axes.set(
        xscale="log",
        ylabel="Log-likelihood in held-out data",
        xlabel="a",
    )
    fig.tight_layout()
    fig.savefig("figures/thermodynamic_model_delta.cv.png", dpi=300)
    # fig.savefig("figures/thermodynamic_model_delta.cv.svg")
