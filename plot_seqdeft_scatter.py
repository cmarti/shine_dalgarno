import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.src.plot.mpl import plot_density_vs_frequency

if __name__ == "__main__":
    inferred = pd.read_csv("data/seqdeft_inference.csv", index_col=0)
    fig, axes = plt.subplots(1, 1, figsize=(3.75, 3.0))

    with np.errstate(divide="ignore"):
        logf = np.log10(inferred["frequency"])
        logq = np.log10(inferred["Q_star"])
        data = pd.DataFrame({"logR": logf, "logQ": logq}).dropna()

    axes.scatter(
        logf,
        logq,
        color="black",
        s=7.5,
        lw=0,
        alpha=0.1,
        zorder=2,
        label="Observed",
    )

    zero_idx = np.isinf(logf)
    finite_idx = np.logical_not(zero_idx)
    zero_counts_logq = logq[zero_idx]
    fake_logf = np.full(zero_counts_logq.shape, logf[finite_idx].min() - 0.5)
    axes.scatter(
        fake_logf,
        zero_counts_logq,
        marker="<",
        color="red",
        s=5,
        alpha=0.05,
        zorder=2,
        label="Unobserved",
    )

    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
    axes.plot(lims, lims, color="grey", linewidth=0.5, alpha=0.5, zorder=1)
    axes.set(
        xlabel=r"$log_{10}$(Frequency)",
        ylabel=r"$log_{10}$(Q*)",
        xlim=lims,
        ylim=lims,
    )
    axes.legend(loc=2, fontsize=9)
    axes.set(aspect="equal")

    fig.tight_layout()
    fig.savefig("figures/seqdeft_scatter.svg")
    fig.savefig("figures/seqdeft_scatter.png")
