import pandas as pd

from gpmap.inference import MinimumEpistasisInterpolator

if __name__ == "__main__":
    print("Loading data")
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

    print("Interpolating missing sequences using MEI")
    model = MinimumEpistasisInterpolator(
        seq_length=9, alphabet_type="rna", P=2
    )
    
    print("Computing MAP for complete sequence-space")
    inferred = model.predict()
    inferred.to_csv("results/inferred_vc_regression.csv")
    
    print("Computing posterior variances for test data")
    model.fit(X=X_train, y=y_train)
    pred = model.predict(X_pred=X_test, calc_variance=True)
    pred.to_csv("results/mei.test.csv")
