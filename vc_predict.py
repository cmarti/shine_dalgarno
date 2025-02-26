import numpy as np
import pandas as pd

from gpmap.src.inference import VCregression


if __name__ == "__main__":
    print('Loading data')
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0)
    test = pd.read_csv("processed/dmsc.test.csv", index_col=0)

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

    lambdas = np.load("results/vc.lambdas.npy")
    model = VCregression(seq_length=9, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X_train, y=y_train, y_var=y_var_train)

    print('Computing MAP for complete sequence-space')
    inferred = model.predict()
    inferred.to_csv("results/inferred_vc_regression.csv")

    print('Computing posterior variances for test data')
    test_pred = model.predict(X_pred=X_test, calc_variance=True)
    test_pred.to_csv("results/SD_test.vc_pred.csv")
