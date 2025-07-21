import numpy as np

from scripts.utils import load_sequence_data
from gpmap.inference import SeqDEFT


if __name__ == "__main__":
    for species in ["b_sub", "e_coli"]:
        print("Computing Maximum a Posteriori from {} genome".format(species))
        print("\tLoading data")
        X = load_sequence_data(species)

        print("\tLoading hyperparameter 'a'")
        a = np.load("results/{}.seqdeft.a.npy".format(species))

        print("\tInferring genotype-phenotype map")
        model = SeqDEFT(P=2, a=a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)
        pred = model.predict()

        print("\tStoring results")
        fpath = "results/{}.seqdeft.map.csv".format(species)
        pred.to_csv(fpath)
