import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    labels = {"ds": "Datashader", "mpl": "Matplotlib"}
    times = pd.read_csv("results/rendering_times.csv", index_col=0)
    times["a"] = [4 if x == "dna" else 20 for x in times["type"]]
    times["n"] = times["a"] ** times["seq_length"]
    times = times.iloc[1:, :]
    print(times)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH * 0.5, FIG_WIDTH * 0.275),
        sharex=True,
        sharey=True,
    )

    palette = {"Datashader": "black", "Matplotlib": "grey"}
    for axes, (alphabet, df) in zip(subplots, times.groupby("type")):
        df = pd.melt(df, id_vars=["seq_length", "type", "a", "n"]).dropna()
        df["backend"] = [labels[x] for x in df["variable"]]
        sns.lineplot(
            x="n",
            y="value",
            hue="backend",
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
            ylabel="Rendering time (s)",
            xscale="log",
            yscale="log",
            title=alphabet.upper(),
        )
        axes.legend(loc=2)

    fig.tight_layout()
    fig.savefig("figures/rendering_times.png", dpi=300)
    fig.savefig("figures/rendering_times.svg", dpi=300)
