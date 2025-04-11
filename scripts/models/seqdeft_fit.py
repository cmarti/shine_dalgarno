import numpy as np

from gpmap.inference import SeqDEFT


if __name__ == "__main__":
    print('Fitting SeqDEFT models')
    for species in ["b_sub", "e_coli"]:
        print("Loading data from {}".format(species))
        fpath = "processed/{}.seqs.txt".format(species)
        X = np.array([line.strip() for line in open(fpath)])
        print("\t{} sequences loaded".format(X.shape[0]))

        print("\tRunning cross-validation")
        model = SeqDEFT(P=2, seq_length=9, alphabet_type="rna")
        model.fit(X=X)

        fpath = "results/{}.seqdeft.cv_results.csv".format(species)
        model.logL_df.to_csv(fpath)
        
        fpath = "results/{}.seqdeft.a.npy".format(species)
        np.save(fpath, model.a)
