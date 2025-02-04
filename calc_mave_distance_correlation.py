import numpy as np
import pandas as pd

from gpmap.src.inference import VCregression


if __name__ == "__main__":
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0)
    X_train, y_train, y_var_train = (
        train.index.values,
        train.y.values,
        train.y_var.values,
    )

    # Compute the empirical correlation-distance function
    model = VCregression(seq_length=9, alphabet_type="rna")
    cov, ns = model.calc_covariance_distance(
        X=X_train, y=y_train - y_train.mean()
    )
    dcor = pd.DataFrame({"d": np.arange(cov.shape[0]), "rho": cov / cov[0]})
    dcor.to_csv("results/dmsc.empirical_distance_correlation.csv")
