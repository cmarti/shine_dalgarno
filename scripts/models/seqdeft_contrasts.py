import numpy as np
import pandas as pd

from gpmap.inference import SeqDEFT
from scripts.utils import get_contrast_matrix, load_sequence_data


if __name__ == "__main__":
    print("Computing posterior distribution for contrasts")
    for species in ["e_coli", "b_sub"]:
        print("Analyzing data from {} genome".format(species))
        X = load_sequence_data(species)

        print("\tLoading hyperparameter 'a'")
        a = np.load("results/{}.seqdeft.a.npy".format(species))
        model = SeqDEFT(P=2, a=a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)

        print("\tLoading MAP sequence distribution")
        seqdeft = pd.read_csv("results/{}.seqdeft.map.csv".format(species))
        obs = pd.DataFrame(
            {
                "freqs": model.likelihood.R,
                "stat_freqs": seqdeft["Q_star"].values,
            },
            index=model.genotypes,
        )
        contrasts_matrix = get_contrast_matrix(seqdeft=True, obs=obs)

        print("Computing the posterior for the contrasts")
        fpath = "results/{}.seqdeft.contrasts.csv".format(species)
        contrasts = model.make_contrasts(contrasts_matrix)
        contrasts.to_csv(fpath)
