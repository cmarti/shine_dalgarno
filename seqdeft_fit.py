import numpy as np

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    print('Loading data')
    X = np.array([line.strip() for line in open("processed/SD_seqs.txt")])

    print('Fitting SeqDEFT hyperparameter through Cross-Validation')
    model = SeqDEFT(P=2, seq_length=9, alphabet_type="rna")
    inferred = model.fit(X=X)

    print('Storing inference results')
    inferred["counts"] = inferred["frequency"] * X.shape[0]
    inferred.to_csv("results/seqdeft_inference.csv")
    model.logL_df.to_csv("results/seqdeft_hyperparam_optimization.csv")
    with open("results/seqdeft_optimal_a.txt", "w") as fhand:
        fhand.write("{}".format(model.a))
