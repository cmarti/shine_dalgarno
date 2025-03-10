import numpy as np

from gpmap.src.inference import SeqDEFT


if __name__ == "__main__":
    for species in ["b_sub"]:  # , "e_coli"]:
        print("Loading data from {}".format(species))
        X = np.array(
            [
                line.strip()
                for line in open("data/{}.SD_seqs.txt".format(species))
            ]
        )
        print("{} sequences loaded".format(X.shape[0]))

        print("Fitting SeqDEFT hyperparameter through Cross-Validation")
        model = SeqDEFT(P=2, seq_length=9, alphabet_type="rna")
        inferred = model.fit(X=X)

        print("Storing inference results")
        inferred["counts"] = inferred["frequency"] * X.shape[0]
        inferred.to_csv("results/{}.seqdeft_inference.csv".format(species))
        model.logL_df.to_csv(
            "results/{}.seqdeft_hyperparam_optimization.csv".format(species)
        )
        with open(
            "results/{}.seqdeft_optimal_a.txt".format(species), "w"
        ) as fhand:
            fhand.write("{}".format(model.a))
