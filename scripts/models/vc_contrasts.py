import numpy as np

from scripts.utils import get_contrast_matrix, load_experimental_data
from gpmap.inference import VCregression


if __name__ == "__main__":
    X, y, y_var = load_experimental_data()[:3]
    
    print("Loading variance components")
    lambdas = np.load("results/vcregression.lambdas.npy")
    model = VCregression(seq_length=9, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X, y=y, y_var=y_var)

    contrasts_matrix = get_contrast_matrix()
    print('Computing the posterior for the contrasts')
    contrasts = model.make_contrasts(contrasts_matrix)
    contrasts.to_csv('results/vcregression.contrasts.csv')

    # # This may be requried now in the plotting script
    # print("Start computation 2")
    # peaks_contrasts = model.make_contrasts(contrasts_matrix)
    # peaks_contrasts["mutation"] = [
    #     x.split("_")[0] for x in peaks_contrasts.index.values
    # ]
    # peaks_contrasts["background"] = [
    #     x.split("_")[-1] for x in peaks_contrasts.index.values
    # ]
    # peaks_contrasts.to_csv("results/vcregression_peaks_contrasts.csv")

