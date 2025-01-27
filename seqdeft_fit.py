import numpy as np

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    X = np.array([line.strip() for line in open("data/SD_seqs.txt")])
    model = SeqDEFT(P=2, seq_length=9, alphabet_type="rna")
    inferred = model.fit(X=X)
    inferred["counts"] = inferred["frequency"] * X.shape[0]

    inferred.to_csv("data/seqdeft_inference.csv")
    model.logL_df.to_csv("data/seqdeft_hyperparam_optimization.csv")
    with open("data/optimal_a.txt", "w") as fhand:
        fhand.write("{}".format(model.a))
