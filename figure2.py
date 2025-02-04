import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from gpmap.src.plot.mpl import plot_hyperparam_cv
from scipy.stats import pearsonr
from plot_utils import FIG_WIDTH


if __name__ == '__main__':
    upstream_bases = 20
    distance_start_codon = 4
    seq_length = 9
    sd_start = upstream_bases - distance_start_codon - seq_length
    x1, x2 = 0.025, 0.55
    y1, y2, y3 = 0.95, 0.65, 0.35

    print('Loading data')
    dcor = pd.read_csv('results/dmsc.empirical_distance_correlation.csv', index_col=0)
    vc = pd.read_csv('results/vc.prior_variance_components.csv', index_col=0)
    train = pd.read_csv('processed/dmsc.train.csv', index_col=0)
    test = pd.read_csv('processed/dmsc.test.csv', index_col=0)

    full = pd.read_csv("results/vcregression.full.csv", index_col=0)
    pred = pd.read_csv('results/vcregression.test.csv', index_col=0)
    test = test.join(pred, rsuffix='_pred')
    train = train.join(full, rsuffix='_pred')

    fig = plt.figure(figsize=(FIG_WIDTH * 0.5, 0.7 * FIG_WIDTH))
    gs = gs.GridSpec(3, 4,
                     width_ratios=[1, 0.05, 1, 0.05])
    
    print('Plotting empirical distance correlation')
    axes = fig.add_subplot(gs[0, 0])
    axes.scatter(dcor['d'], dcor['rho'], color='black', s=10)
    axes.plot(dcor['d'], dcor['rho'], color='black', lw=1)
    axes.set(ylabel='Empirical correlation', ylim=(None, 1.05),
            xlabel='Hamming distance', xticks=np.arange(dcor.shape[0]))
    axes.axhline(0, linestyle='--', color='grey', lw=0.75)
    axes.text(x1, y1, 'A', fontsize=10, weight='bold', transform=fig.transFigure)

    print('Plotting prior variance components')
    axes = fig.add_subplot(gs[0, 2])
    axes.bar(x=vc['k'], height=vc['variance_perc'], color='black')
    axes.set(xlabel='Interaction order $k$', ylabel='% variance explained',
            xticks=np.arange(1, 10), ylim=(0, 52.5))

    axes = axes.twinx()
    axes.scatter(vc['k'], np.cumsum(vc['variance_perc']), color='grey', s=10)
    axes.plot(vc['k'], np.cumsum(vc['variance_perc']), color='grey', lw=1)
    axes.set(ylabel='% cumulative variance', ylim=(0, 105))
    axes.text(x2, y1, 'B', fontsize=10, weight='bold', transform=fig.transFigure)

    print('Plotting predicted vs. observed in training set')
    axes = fig.add_subplot(gs[1, 0])
    cbar_axes = fig.add_subplot(gs[1, 1])
    lims = (full['y'].min()-0.5, full['y'].max() + 0.5)
    bins = np.linspace(lims[0], lims[-1], 100)
    x, y = train['y_pred'], train['y']
    r2 = pearsonr(x, y)[0] ** 2
    sns.histplot(x=x, y=y, cmap='viridis', ax=axes, bins=(bins, bins), 
                 cbar_ax=cbar_axes,
                 cbar=True, cbar_kws={'label': 'Number of sequences', 'shrink': 0.95})
    axes.set(xlim=lims, ylim=lims, xticks=np.arange(4),
            xlabel='Training predicted log(GFP)', ylabel='Training measured log(GFP)')
    axes.axline((0, 0), (1, 1), lw=0.5, c='grey', linestyle='--')
    axes.text(0.05, 0.95, '$R^2$' + '={:.2f}'.format(r2),
            transform=axes.transAxes, va='top', ha='left', fontsize=6)
    axes.text(x1, y2, 'C', fontsize=10, weight='bold', transform=fig.transFigure)

    print('Plotting predicted vs. observed in test set')
    axes = fig.add_subplot(gs[1, 2])
    x, y, yerr = test['y_pred'], test['y'], 2 * test['std']
    r2 = pearsonr(y, x)[0] ** 2
    calibration = np.logical_and(y < test['ci_95_upper'],
                                 y > test['ci_95_lower']).mean()
    axes.errorbar(x, y, yerr=yerr,
                color='black', fmt='o', alpha=0.2, markeredgewidth=0,
                markersize=2.5, lw=.5)
    axes.set(xlim=lims, ylim=lims, xticks=np.arange(4),
            ylabel='Test predicted log(GFP)', xlabel='Test measured log(GFP)')
    axes.axline((0, 0), (1, 1), lw=0.5, c='grey', linestyle='--')
    axes.text(0.05, 0.97, '$R^2$' + '={:.2f}\nCalibration={:.1f}%\nn={}'.format(r2, calibration*100, test.shape[0]),
            transform=axes.transAxes, va='top', ha='left', fontsize=6)
    axes.text(x2, y2, 'D', fontsize=10, weight='bold', transform=fig.transFigure)


    fig.subplots_adjust(bottom=0.175, left=0.15, hspace=0.4, wspace=0.5, top=0.925, right=0.95)
    pos = cbar_axes.get_position()
    pos.p0[0] -= 0.05 
    pos.p1[0] -= 0.05 
    print(pos.p0)
    cbar_axes.set_position(pos)
    # fig.tight_layout()
    fig.savefig('figures/figure2.png', dpi=300)
