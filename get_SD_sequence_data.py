import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
import matplotlib.pyplot as plt
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    upstream_bases = 20
    seq_length = 9

    for species, distance_start_codon in [("e_coli", 4), ("b_sub", 6)]:
        gene_data = pd.read_csv(
            "processed/{}.gene_5utr.csv".format(species), index_col=0
        )
        start = upstream_bases - distance_start_codon - seq_length
        end = start + seq_length

        with open("processed/{}.SD_seqs.txt".format(species), "w") as fhand:
            for seq in gene_data["background"]:
                if "N" in seq:
                    continue
                fhand.write(seq[start:end] + "\n")
