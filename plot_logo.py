from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
import logomaker as lm

from settings import SD_SORTSEQ_DIR, SD_BACKGROUNDS, PLOTS_DIR, SD_RIBOSEQ_DIR,\
    SD_DIR
from utils import LogTrack
from gpmap.visualization import Visualization
from plot_utils import savefig, init_fig


if __name__ == '__main__':
    fpath = join(SD_DIR, 'seqdeft', 'sd_seq_genes.csv.gz')
    data = pd.read_csv(fpath, index_col=0)
    data = data.loc[data['species'] == 'e_coli'].set_index('gene_id')
    
    freqs = lm.alignment_to_matrix(data['SD'], to_type='probability')
    freqs.index = np.arange(1, freqs.shape[0] + 1) - 14
    
    fig, subplots = init_fig(2, 1, colsize=4, rowsize=1.8)
    
    axes = subplots[0]
    lm.Logo(freqs, ax=axes)
    axes.set(ylabel='Nucleotide frequency', xticks=freqs.index,
             xticklabels=[])
    
    info = lm.transform_matrix(freqs, from_type='probability', to_type='information')
    
    axes = subplots[1]
    lm.Logo(info, ax=axes)
    axes.set(ylabel='Information', xticks=info.index,
             xlabel='Relative position to Start codon')
    
    sns.despine()
    
    savefig(fig, fname='e_coli_sd_logo')
