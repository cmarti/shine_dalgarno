import pandas as pd
import numpy as np

from itertools import combinations, product

from gpmap.inference import SeqDEFT


if __name__ == "__main__":
    print('Defining contrasts to make')
    bc1 = np.array(["".join(x) for x in product("ACGU", repeat=3)])
    bc2 = np.array(["".join(x) for x in product("ACGU", repeat=6)])
    bc3 = np.array(["".join(x) for x in product("ACGU", repeat=9)])
    bc4 = np.array(["".join(x) for x in product("ACGU", repeat=2)])
    p1, p2, p3 = 1.0 / bc1.shape[0], 1.0 / bc2.shape[0], 1.0 / bc3.shape[0]
    p4 = 1.0 / bc4.shape[0]
    contrasts = {
        "AGGAGGNNN": {"AGGAGG{}".format(x): p1 for x in bc1},
        "NGGAGGAGN": {"{}GGAGGAG{}".format(x[0], x[-1]): p4 for x in bc4},
        "NNNAGGNNN": {"{}AGG{}".format(x[:3], x[3:]): p2 for x in bc2},
        "NNNAGGAGG": {"{}AGGAGG".format(x): p1 for x in bc1},
        "NNNNNNNNN": {x: p3 for x in bc3},
    }
    contrasts_matrix1 = pd.DataFrame(contrasts).fillna(0)
    for col in contrasts_matrix1.columns:
        contrasts_matrix1[col] -= contrasts_matrix1["NNNNNNNNN"]
    contrasts_matrix1.drop("NNNNNNNNN", axis=1, inplace=True)
    
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
    contrasts_matrix2 = pd.DataFrame(contrasts).fillna(0)

    print('Computing posterior distribution for contrasts')
    for species in ["b_sub", "e_coli"]:
        print('Analyzing data from {} genome'.format(species))
        
        print("\tLoading data")
        fpath = "processed/{}.SD_seqs.txt".format(species)
        X = np.array([line.strip() for line in open(fpath)])

        print("\tLoading hyperparameter")
        fpath = "results/{}.seqdeft_optimal_a.txt".format(species)
        with open(fpath, "r") as fhand:
            optimal_a = float([line.strip() for line in fhand][0])

        print("\tEstimating posterior along the 3-nucleotide shift path")
        model = SeqDEFT(P=2, a=optimal_a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)
        pred = model.make_contrasts(contrasts_matrix1)
        pred.to_csv("results/{}.seqdeft_path_contrasts.csv".format(species))

        print("\tEstimating posterior of mutational effects across shifts")
        peaks_contrasts = model.make_contrasts(contrasts_matrix2)
        peaks_contrasts["mutation"] = [
            x.split("_")[0] for x in peaks_contrasts.index.values
        ]
        peaks_contrasts["background"] = [
            x.split("_")[-1] for x in peaks_contrasts.index.values
        ]
        peaks_contrasts.to_csv(
            "results/{}.seqdeft_peaks_contrasts.csv".format(species)
        )
