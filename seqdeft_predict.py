import numpy as np

from gpmap.inference import SeqDEFT


if __name__ == "__main__":
    print("Computing Maximum a Posteriori")
    for species in ["b_sub", "e_coli"]:
        print("Analyzing data from {} genome".format(species))

        print("\tLoading data")
        fpath = "processed/{}.SD_seqs.txt".format(species)
        X = np.array([line.strip() for line in open(fpath)])

        print("\tLoading hyperparameter")
        with open("data/optimal_a.txt", "r") as fhand:
            optimal_a = float([line.strip() for line in fhand][0])

        print("\tInferring genotype-phenotype map")
        model = SeqDEFT(P=2, a=optimal_a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)
        pred = model.predict()

        print("\tStoring results")
        fpath = "results/{}.seqdeft_inference.csv".format(species)
        pred.to_csv(fpath)
