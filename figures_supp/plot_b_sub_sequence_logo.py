import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
import matplotlib.pyplot as plt
from plot_utils import FIG_WIDTH


if __name__ == "__main__":
    upstream_bases = 20
    seq_length = 9

    species, distance_start_codon = "b_sub", 6
    sd_start = upstream_bases - distance_start_codon - seq_length

    print("Loading data")
    gene_data = pd.read_csv(
        "data/{}.gene_5utr.csv".format(species), index_col=0
    )
    background_seqs = [
        x[: upstream_bases + 3] for x in gene_data["background"]
    ]
    print(len(background_seqs[0]))

    print("Computing allele frequencies")
    allele_freqs = logomaker.alignment_to_matrix(
        background_seqs, to_type="probability", pseudocount=0
    )

    print("Plotting logo")
    fig, axes = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * 0.6, FIG_WIDTH * 0.14)
    )
    logo = logomaker.Logo(allele_freqs, ax=axes, color_scheme="classic")
    for p in list(range(sd_start)) + list(range(sd_start + seq_length, 20)):
        for c in "AUGC":
            logo.style_single_glyph(p, c, alpha=0.3)

    xticks = np.arange(allele_freqs.shape[0])
    xticklabels = ["-{}".format(x) for x in range(1, 21)[::-1]] + [
        "+1",
        "+2",
        "+3",
    ]
    axes.set(
        ylabel="Frequency",
        xlabel="Position relative to start codon",
        xticks=xticks,
        xticklabels=xticklabels,
        yticks=[0, 0.5, 1],
    )
    fig.tight_layout()
    fname = "{}.5utr_sequence_logo".format(species)
    fig.savefig("figures/{}.png".format(fname), dpi=300)
    fig.savefig("figures/{}.svg".format(fname), dpi=300)
