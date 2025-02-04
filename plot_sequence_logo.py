import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
import matplotlib.pyplot as plt
from plot_utils import FIG_WIDTH


if __name__ == '__main__':
    upstream_bases = 20
    distance_start_codon = 4
    seq_length = 9
    sd_start = upstream_bases - distance_start_codon - seq_length

    print('Loading data')
    gene_data = pd.read_csv('processed/gene_data.csv', index_col=0)
    background_seqs = [x[:-upstream_bases] for x in gene_data['background']]

    print('Computing allele frequencies')
    allele_freqs = logomaker.alignment_to_matrix(background_seqs, to_type='probability', pseudocount=0)

    print('Plotting logo')
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.5, FIG_WIDTH * 0.135))
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
    fig.tight_layout()
    fig.savefig('figures/SD_sequence_logo.png', dpi=300)
    # fig.savefig('figures/SD_sequence_logo.svg', dpi=300)