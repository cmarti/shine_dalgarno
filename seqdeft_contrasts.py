import pandas as pd
import numpy as np

from itertools import combinations

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    for species in ["b_sub"]:
        X = np.array(
            [
                line.strip()
                for line in open("processed/{}.SD_seqs.txt".format(species))
            ]
        )

        with open(
            "results/{}.seqdeft_optimal_a.txt".format(species), "r"
        ) as fhand:
            optimal_a = float([line.strip() for line in fhand][0])

        model = SeqDEFT(P=2, a=optimal_a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)

        print("Start computation 1")
        seqs = ["AGGAGAAUA", "AGGAGGAGA", "UUAAGAAUA", "UUAAGGAGC"]
        ref = "UAAGGAGCA"
        contrasts = {}
        for seq in seqs:
            contrasts[seq] = {ref: -1, seq: 1}
        contrasts_matrix = pd.DataFrame(contrasts).fillna(0)

        pred = model.make_contrasts(contrasts_matrix)
        pred.to_csv("results/{}.seqdeft_path_contrasts.csv".format(species))

        backgrounds = ["UUAAGGAGC", "UAAGGAGCA", "AAGGAGCAG"]
        positions = np.arange(-13, -4)
        contrasts = {}
        for bc1, bc2 in combinations(backgrounds, 2):
            for p, (pos, a1, a2) in enumerate(zip(positions, bc1, bc2)):
                if a1 == a2:
                    continue
                label = "{}{}{}".format(a1, pos, a2)

                for bc in [bc1, bc2]:
                    s = [c for c in bc]
                    s[p] = a1
                    s1 = "".join(s)
                    s[p] = a2
                    s2 = "".join(s)
                    contrasts["{}_in_{}".format(label, bc)] = {s1: -1, s2: 1}
        contrasts_matrix = pd.DataFrame(contrasts).fillna(0)

        print("Start computation 2")
        peaks_contrasts = model.make_contrasts(contrasts_matrix)
        peaks_contrasts["mutation"] = [
            x.split("_")[0] for x in peaks_contrasts.index.values
        ]
        peaks_contrasts["background"] = [
            x.split("_")[-1] for x in peaks_contrasts.index.values
        ]
        peaks_contrasts.to_csv(
            "results/{}seqdeft_peaks_contrasts.csv".format(species)
        )
