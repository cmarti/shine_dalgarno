import numpy as np

from gpmap.inference import SeqDEFT


if __name__ == "__main__":
    print('Fitting SeqDEFT models')
    for species in ["b_sub", "e_coli"]:
        print("Loading data from {}".format(species))
        fpath = "data/{}.SD_seqs.txt".format(species)
        X = np.array([line.strip() for line in open(fpath)])
        print("\n{} sequences loaded".format(X.shape[0]))

        print("\nRunning cross-validation")
        model = SeqDEFT(P=2, seq_length=9, alphabet_type="rna")
        model.fit(X=X)

        fpath = "results/{}.seqdeft_hyperparam_optimization.csv".format(species)
        model.logL_df.to_csv(fpath)
        
        fpath = "results/{}.seqdeft_optimal_a.txt".format(species)
        with open(fpath, "w") as fhand:
            fhand.write("{}".format(model.a))
