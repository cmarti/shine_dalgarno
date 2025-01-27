import itertools
import sys
import time
from time import ctime
from _pickle import dump, load

import numpy as np
import scipy.stats as stats

from settings import NUCLEOTIDES, COMPLEMENT, CONSTRAINTS_BASES, ALLOWED_BP,\
    RNA_ALPHABET


def logit(p):
    return(np.log(p  /(1 - p)))


def invlogit(x):
    return(np.exp(x) / (1  + np.exp(x)))


class LogTrack(object):
    '''Logger class'''

    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = '[ {} ] {}\n'.format(ctime(), msg)
        else:
            msg += '\n'
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write('Finished succesfully. Time elapsed: {:.1f} s'.format(t))


def write_log(log, msg):
    if log is not None:
        log.write(msg)


def generate_possible_sequences(l, alphabet=RNA_ALPHABET):
    for seq in itertools.product(alphabet, repeat=l):
        yield(''.join(seq))


def reverse_complement(seq):
    return(''.join(COMPLEMENT.get(x, x) for x in seq[::-1]))


def get_random_seq(length):
    return(''.join(np.random.choice(NUCLEOTIDES, size=length)))


def add_random_flanks(seq, length, only_upstream=False):
    if only_upstream:
        flank = get_random_seq(length)
        new_seq = flank + seq
    else:
        flanks = get_random_seq(2 * length)
        new_seq = flanks[:length] + seq + flanks[length:]
    return(new_seq)


def write_landscape(landscape_iter, fpath, verbose=False):
    with open(fpath, 'w') as fhand:
        fhand.write('sequence\tdG\tKa\tP_helix\n')
        for i, (seq, dG, Ka, p_helix) in enumerate(landscape_iter):
            fhand.write('\t'.join([seq, str(dG), str(Ka), str(p_helix)]) + '\n')
            if verbose and i % 10000 == 0:
                total_seqs = len(NUCLEOTIDES) ** len(seq)
                print('Sequences processed: {} out of {}'.format(i, total_seqs)) 


def get_constraints_idx(constraints):
    c1, c2 = constraints
    idx1 = [i for i, x in enumerate(c1) if x == '(']
    idx2 = [i for i, x in enumerate(c2[::-1]) if x == ')']
    return(idx1, idx2)


def get_constraints(max_flapping, seq_len):
    # Full unconstrained ensembl
    yield(('.' * seq_len, '.' * seq_len))
    
    # Add bulge
    constrain1 = '((x((((('
    constrain2 = '.)))))))'
    yield((constrain1, constrain2))
    
    constrain1 = '((x((((x'
    constrain2 = '.x))))))'
    yield((constrain1, constrain2))
    
    # Get flapping ends
    for i in range(max_flapping + 1):
        constrain1 = 'x' * i + '(' * (seq_len - i)
        constrain2 = ')' * (seq_len - i) + 'x' * i
        yield((constrain1, constrain2))
        
        if i > 0:
            constrain1 = '(' * (seq_len - i) + 'x' * i
            constrain2 = 'x' * i + ')' * (seq_len - i)
            yield((constrain1, constrain2))
        
            # Get flapping ends at the other side
            opposite_flapping = max_flapping - i
            if opposite_flapping > 0 :
                for j in range(1, opposite_flapping + 1):
                    constrain1 = 'x' * i + '(' * (seq_len - i - j) + 'x' * j
                    constrain2 = 'x' * j + ')' * (seq_len - i - j) + 'x' * i
                    yield((constrain1, constrain2))


def embed_constraint(constraint, flank_length):
    flank = '.' * flank_length
    return(flank + constraint[0] + flank + '&' + constraint[1])


def get_full_constraints(max_flapping, seq_len, flank_length):
    flank = '.' * flank_length
    for constrain1, constrain2 in get_constraints(max_flapping, seq_len):
        yield(flank + constrain1 + flank + '&' + constrain2)


def is_valid_seq(seq1, seq2, constraints):
    if constraints[0] != '.' * len(seq1):
        idx1, idx2 = get_constraints_idx(constraints)
        for i, j in zip(idx1, idx2):
            b1, b2 = seq1[i], seq2[::-1][j]
            if (b1, b2) not in ALLOWED_BP:
                return(False)
    return(True)
        

def get_seq(genome, chrom, start, end, strand):
    seq = genome.fetch(chrom, start, end)
    if strand == '-':
        seq = reverse_complement(seq)
    
    return(seq.upper())


def load_pickle(fpath):
    with open(fpath, 'rb') as fhand:
        data = load(fhand)
    return(data)


def write_pickle(data, fpath):
    with open(fpath, 'wb') as fhand:
        dump(data, fhand)
    return(data)


def get_single_mutants(seqs):
    new_seqs = set()
    for seq in seqs:
        for i, nc in enumerate(seq):
            for nc2 in 'AGCU':
                if nc2 == nc:
                    continue
                new_seqs.add(seq[:i] + nc2 + seq[i+1:])
    return(new_seqs)


def get_mutants(seqs, max_d=2):
    seqs = set(seqs)
    
    for _ in range(0, max_d):
        seqs = seqs.union(get_single_mutants(seqs))
    return(seqs)


def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
      
    from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
