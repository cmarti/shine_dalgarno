import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    times = pd.read_csv("results/times_P_U_operator.csv", index_col=0)
    times["n"] = times["n_alleles"] ** times["seq_length"]

    fig, subplots = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.475),
        sharey="col",
    )

    palette = {"Linear Operator": "black", "Dense matrix": "grey"}
    variables = {
        "matvec_time": "Matrix-vector product time (s)",
        "overhead_time": r"$P_U$ building time (s)",
        "current_memory": "Memory usage (MB)",
    }
    for axes_row, (alphabet, df) in zip(subplots, times.groupby("type")):
        for axes, (y, ylabel) in zip(axes_row, variables.items()):
            sns.lineplot(
                x="n",
                y=y,
                hue="operator",
                data=df,
                ax=axes,
                palette=palette,
                errorbar="sd",
                err_style="bars",
                err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
                lw=0.75,
            )
            axes.set(
                xlabel="Genotype-phenotype map size",
                ylabel=ylabel,
                xscale="log",
                yscale="log",
            )
            axes.legend_.set_visible(False)
            axes.text(
                0.05,
                0.95,
                alphabet.upper(),
                transform=axes.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
    subplots[0][0].legend(loc=4)

    fig.tight_layout()
    fig.savefig("figures/times_P_U_operator.png", dpi=300)
    fig.savefig("figures/times_P_U_operator.svg", dpi=300)
