import numpy as np

from gpmap.inference import SeqDEFT
from scripts.utils import get_contrast_matrix, load_sequence_data


if __name__ == "__main__":
    contrasts_matrix = get_contrast_matrix(seqdeft=True)

    print("Computing posterior distribution for contrasts")
    for species in ["b_sub", "e_coli"]:
        print("Analyzing data from {} genome".format(species))
        X = load_sequence_data(species)

        print("\tLoading hyperparameter 'a'")
        a = np.load("results/{}.seqdeft.a.npy".format(species))
        model = SeqDEFT(P=2, a=a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)

        print("Computing the posterior for the contrasts")
        fpath = "results/{}.seqdeft.contrasts.csv".format(species)
        contrasts = model.make_contrasts(contrasts_matrix)
        contrasts.to_csv(fpath)
