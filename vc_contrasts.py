import pandas as pd
import numpy as np

from itertools import combinations, product

from gpmap.inference import VCregression


if __name__ == "__main__":
    print("Loading data")
    data = pd.read_csv("processed/dmsc.train.csv", index_col=0)

    X, y, y_var = (
        data.index.values,
        data.y.values,
        data.y_var.values,
    )
    lambdas = np.load("results/vc.lambdas.npy")
    model = VCregression(seq_length=9, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X, y=y, y_var=y_var)

    print("Start computation 1")
    seqs = ["AGGAGAAUA", "AGGAGGAGA", "UUAAGAAUA", "UUAAGGAGC"]
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
        "AAGGAGGUG": {"AAGGAGGUG": 1.0},
    }
    contrasts_matrix = pd.DataFrame(contrasts).fillna(0)
    pred = model.make_contrasts(contrasts_matrix)
    pred.to_csv("results/vcregression_path_contrasts.csv")
    print(pred)
    exit()

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
    peaks_contrasts.to_csv("results/vcregression_peaks_contrasts.csv")
