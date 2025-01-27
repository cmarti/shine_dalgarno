import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.src.plot.mpl import plot_hyperparam_cv

if __name__ == "__main__":
    logL_df = pd.read_csv(
        "data/seqdeft_hyperparam_optimization.csv", index_col=0
    )
    logL_df = logL_df.loc[logL_df["a"] > 0, :]
    fig, axes = plt.subplots(1, 1, figsize=(3.75, 3.0))

    plot_hyperparam_cv(logL_df, axes, show_folds=False)
    axes.set(xticks=np.arange(1, 7))
    axes.legend(loc=4, fontsize=9)

    fig.tight_layout()
    fig.savefig("figures/seqdeft_hyperparameter_opt.svg")
    fig.savefig("figures/seqdeft_hyperparameter_opt.png")
