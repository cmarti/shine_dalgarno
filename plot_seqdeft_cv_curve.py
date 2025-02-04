import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.src.plot.mpl import plot_hyperparam_cv
from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    logL_df = pd.read_csv(
        "results/seqdeft_hyperparam_optimization.csv", index_col=0
    )
    logL_df = logL_df.loc[logL_df["a"] > 0, :]
    max_logL = logL_df.groupby('log_a')['logL'].mean().max()
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.25, FIG_WIDTH * 0.225))

    plot_hyperparam_cv(logL_df, axes, show_folds=False, size=2)
    axes.axhline(max_logL, lw=0.5, linestyle='--', color='red')
    axes.set(xticks=np.arange(1, 7))
    legend = axes.legend(loc=4)  
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_alpha(0.5)

    fig.tight_layout()
    fig.savefig("figures/seqdeft_hyperparameter_opt.svg")
    fig.savefig("figures/seqdeft_hyperparameter_opt.png", dpi=300)
