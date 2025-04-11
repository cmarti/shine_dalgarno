import numpy as np

from gpmap.inference import SeqDEFT
from scripts.utils import get_contrast_matrix, load_sequence_data


if __name__ == "__main__":
    contrasts_matrix = get_contrast_matrix()

    print('Computing posterior distribution for contrasts')
    for species in ["b_sub", "e_coli"]:
        print('Analyzing data from {} genome'.format(species))
        
        print("\tLoading data")
        X = load_sequence_data(species)

        print("\tLoading hyperparameter 'a'")
        a = np.load("results/{}.seqdeft.a.npy".format(species))

        print("\tEstimating posterior for contrasts")
        model = SeqDEFT(P=2, a=a, seq_length=9, alphabet_type="rna")
        model.set_data(X=X)
        contrasts = model.make_contrasts(contrasts_matrix)
        contrasts_matrix.to_csv("results/{}.seqdeft.contrasts.csv".format(species))

        # print("\tEstimating posterior of mutational effects across shifts")
        # peaks_contrasts = model.make_contrasts(contrasts_matrix2)
        # peaks_contrasts["mutation"] = [
        #     x.split("_")[0] for x in peaks_contrasts.index.values
        # ]
        # peaks_contrasts["background"] = [
        #     x.split("_")[-1] for x in peaks_contrasts.index.values
        # ]
        # peaks_contrasts.to_csv(
        #     "results/{}.seqdeft_peaks_contrasts.csv".format(species)
        # )
