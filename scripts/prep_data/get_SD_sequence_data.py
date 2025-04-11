import pandas as pd


if __name__ == "__main__":
    upstream_bases = 20
    seq_length = 9

    for species, distance_start_codon in [("e_coli", 4), ("b_sub", 6)]:
        print("Extracting Shine-Dalgarno sequence from {}".format(species))
        fpath = "processed/{}.gene_5utr.csv".format(species)
        gene_data = pd.read_csv(fpath, index_col=0)
        start = upstream_bases - distance_start_codon - seq_length
        end = start + seq_length

        fpath = "processed/{}.seqs.txt".format(species)
        with open(fpath, "w") as fhand:
            for seq in gene_data["background"]:
                if "N" in seq:
                    continue
                fhand.write(seq[start:end] + "\n")
