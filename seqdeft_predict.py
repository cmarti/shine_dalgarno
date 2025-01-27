import pandas as pd
import numpy as np

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    X = np.array([line.strip() for line in open("data/SD_seqs.txt")])
    test = pd.read_csv("data/SD_test_pred.csv", index_col=0)
    X_test = test.index.values
    print(test)
    exit()
    X_test = [""]

    with open("data/optimal_a.txt", "r") as fhand:
        optimal_a = float([line.strip() for line in fhand][0])

    model = SeqDEFT(P=2, a=optimal_a, seq_length=9, alphabet_type="rna")
    model.set_data(X=X)

    print("Start computation")
    test_pred = model.predict(X_pred=X_test, calc_variance=True)
    test_pred.to_csv("data/seqdeft_test_pred.csv")
