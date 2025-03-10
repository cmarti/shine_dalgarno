import numpy as np
import pandas as pd

from gpmap.src.inference import VCregression, MinimumEpistasisInterpolator

if __name__ == "__main__":
    data = pd.read_csv("processed/dmsc.csv", index_col=0)

    np.random.seed(0)

    results = []
    for p in np.geomspace(0.005, 0.95, 20):
        n_train = int(p * data.shape[0])
        for _ in range(3):
            train_idx = np.random.choice(
                data.index, size=n_train, replace=False
            )
            train = data.loc[train_idx, :]
            test_idx = ~np.isin(data.index, train_idx)
            test = data.loc[test_idx, :]

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

            # Make predictions using the VC regression model
            model = VCregression(seq_length=9, alphabet_type="rna")
            model.fit(X=X_train, y=y_train, y_var=y_var_train)
            y_test_pred = model.predict(X_pred=X_test).values.flatten()
            r2 = np.corrcoef(y_test, y_test_pred)[0, 1] ** 2
            rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            record = {"p": p, "r2": r2, "rmse": rmse, "model": "VC"}
            print(record)
            results.append(record)

            # Make predictions using the minimum epistasis interpolator
            model = MinimumEpistasisInterpolator(
                P=2, seq_length=9, alphabet_type="rna"
            )
            model.set_data(X=X_train, y=y_train)
            y_test_pred = model.predict().loc[X_test, :].values.flatten()
            r2 = np.corrcoef(y_test, y_test_pred)[0, 1] ** 2
            rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            record = {"p": p, "r2": r2, "rmse": rmse, "model": "MEI"}
            print(record)
            results.append(record)

    results = pd.DataFrame(results)
    results.to_csv("results/r2.csv")
