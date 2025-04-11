import numpy as np

from scripts.utils import load_experimental_data
from gpmap.inference import VCregression


if __name__ == "__main__":
    X, y, y_var = load_experimental_data()[:3]

    print('Estimating variance components')
    model = VCregression(seq_length=9, alphabet_type="rna")
    model.fit(X, y, y_var)
    np.save("results/vcregression.lambdas.npy", model.lambdas)

    # Save variance components
    print('Saving variance components')
    vc = model.get_variance_components()
    vc.to_csv("results/vcregression.variance_components.csv")
