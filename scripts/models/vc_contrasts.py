import numpy as np

from gpmap.inference import VCregression
from scripts.utils import get_contrast_matrix, load_experimental_data


if __name__ == "__main__":
    contrasts_matrix = get_contrast_matrix()
    X, y, y_var = load_experimental_data()[:3]

    print("Loading variance components")
    lambdas = np.load("results/vcregression.lambdas.npy")
    model = VCregression(seq_length=9, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X, y=y, y_var=y_var)

    print("Computing the posterior for the contrasts")
    contrasts = model.make_contrasts(contrasts_matrix)
    contrasts.to_csv("results/vcregression.contrasts.csv")
