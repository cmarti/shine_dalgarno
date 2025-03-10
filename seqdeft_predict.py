import pandas as pd
import numpy as np

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    for species in ["b_sub"]:  # . 'e_coli]:
        print("Loading data from {}".format(species))
        X = np.array(
            [
                line.strip()
                for line in open("processed/{}.SD_seqs.txt".format(species))
            ]
        )

        with open("data/optimal_a.txt", "r") as fhand:
            optimal_a = float([line.strip() for line in fhand][0])

        model = SeqDEFT(P=2, a=optimal_a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)

        print("Start computation")
        X_test = pd.read_csv("data/SD_test_pred.csv", index_col=0).index.values
        test_pred = model.predict(X_pred=X_test, calc_variance=True)
        test_pred.to_csv("data/{}.seqdeft_test_pred.csv".format(species))
