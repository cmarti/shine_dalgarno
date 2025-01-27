import numpy as np
import pandas as pd

from gpmap.src.inference import VCregression


if __name__ == "__main__":
    train = pd.read_csv("data/dmsc.train.csv", index_col=0)
    test = pd.read_csv("data/dmsc.test.csv", index_col=0)

    X_train, y_train, y_var_train = (
        train.index.values,
        train.y.values,
        train.y_var.values,
    )
    X_test, y_test, y_var_test = (
        test.index.values,
        test.y.values,
        test.y_var.values,
    )

    # Compute the empirical correlation-distance function
    lambdas = np.load("data/vc.lambdas.npy")
    model = VCregression(seq_length=9, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X_train, y=y_train, y_var=y_var_train)
    inferred = model.predict()
    inferred.to_csv("data/inferred_vc_regression.csv")

    test_pred = model.predict(X_pred=X_test, calc_variance=True)
    test_pred.to_csv("data/SD_test.vc_pred.csv")
