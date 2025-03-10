import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from gpmap.src.plot.mpl import plot_hyperparam_cv

from plot_utils import FIG_WIDTH


if __name__ == '__main__':
    upstream_bases = 20
    distance_start_codon = 4
    seq_length = 9
    sd_start = upstream_bases - distance_start_codon - seq_length

    print('Loading data')
    gene_data = pd.read_csv('processed/gene_data.csv', index_col=0)
    inferred = pd.read_csv("results/seqdeft.full.csv", index_col=0)
    inferred["logQ"] = np.log10(inferred["Q_star"])
    max_counts = int(inferred["counts"].max())
    background_seqs = [x[:-upstream_bases] for x in gene_data['background']]
    allele_freqs = logomaker.alignment_to_matrix(background_seqs, to_type='probability', pseudocount=0)
    n_hist = max_counts + 1

    logL_df = pd.read_csv(
        "results/seqdeft.hyperparam_optimization.csv", index_col=0
    )
    logL_df = logL_df.loc[logL_df["a"] > 0, :]
    max_logL = logL_df.groupby('log_a')['logL'].mean().max()

    fig = plt.figure(figsize=(FIG_WIDTH * 0.5, FIG_WIDTH * 0.35))
    nspace = 4
    gs = gs.GridSpec(1 + nspace + n_hist, 2,
                     height_ratios=[0.75] + [3/n_hist] * (n_hist + nspace),
                     width_ratios=[1, 1])
    
    print('Plotting sequence logo')
    axes = fig.add_subplot(gs[0, :])
    logo = logomaker.Logo(allele_freqs, ax=axes, color_scheme='classic')
    for p in list(range(sd_start)) + list(range(sd_start + seq_length, 20)):
        for c in 'AUGC':
            logo.style_single_glyph(p, c, alpha=0.3)

    xticks = np.arange(allele_freqs.shape[0])
    xticklabels = ['-{}'.format(x) for x in range(1, 21)[::-1]] + ['+1', '+2', '+3']
    axes.set(ylabel='Frequency',
            xlabel='Position relative to start codon',
            xticks=xticks, xticklabels=xticklabels,
            yticks=[0, 0.5, 1])
    axes.text(0.05, 0.95, 'A', fontsize=10, weight='bold', transform=fig.transFigure)
    
    print('Plotting CV curve')
    axes = fig.add_subplot(gs[nspace+1:, 0])
    plot_hyperparam_cv(logL_df, axes, show_folds=False, size=2)
    axes.axhline(max_logL, lw=0.5, linestyle='--', color='red')
    axes.set(xticks=np.arange(1, 7))
    legend = axes.legend(loc=4)  
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_alpha(0.5)
    axes.text(0.05, 0.65, 'B', fontsize=10, weight='bold', transform=fig.transFigure)

    print('Plotting histograms')

    bins = np.linspace(inferred["logQ"].min(), inferred["logQ"].max(), 50)
    subplots = [fig.add_subplot(gs[nspace + 1 + i, 1]) for i in range(n_hist)]
    xticks = np.arange(-8, -2)
    for axes, (c, df) in zip(subplots, inferred.groupby("counts")):

        sns.histplot(
            df["logQ"].values,
            bins=bins,
            ax=axes,
            stat="percent",
            color="grey",
            alpha=0.8,
            lw=0,
            label="Estimated probability",
        )
        sns.despine(ax=axes, bottom=False, left=True)
        ylim = axes.get_ylim()
        axes.axvline(
            np.log10(df["frequency"][0]),
            color="black",
            lw=0.75,
            linestyle="-",
            label="Observed frequency",
        )
        axes.set(ylabel="", xlabel='', yticklabels=[], yticks=[], xticklabels=[],
                 xticks=xticks)
        label = "N$_i$={} (n={})".format(int(np.round(c, 0)), df.shape[0])
        if c == 0:
            label = "N$_i$=0"
        axes.text(
            0.01,
            0.95,
            label,
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )

    sns.despine(ax=axes, bottom=False)
    xticklabels = [
        "$10^{-8}$",
        "$10^{-7}$",
        "$10^{-6}$",
        "$10^{-5}$",
        "$10^{-4}$",
        "$10^{-3}$",
    ]
    axes.set(
        xticks=xticks, xticklabels=xticklabels, xlabel="Sequence probability"
    )

    subplots[5].text(-0.075, 5.5, '# sequences', 
                     fontsize=7, rotation=90,
                     ha='center', va='center',
                     transform=axes.transAxes)
    subplots[0].text(0.565, 0.65, 'C', fontsize=10, weight='bold', transform=fig.transFigure)

    fig.subplots_adjust(bottom=0.175, left=0.2, hspace=0.15, wspace=0.25, top=0.925, right=0.95)
    fig.savefig('figures/figure1.png', dpi=300)
