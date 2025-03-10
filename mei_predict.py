import pandas as pd
import numpy as np

from gpmap.src.inference import MinimumEpistasisInterpolator
from gpmap.src.linop import DeltaPOperator
from gpmap.src.matrix import quad

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
    model = MinimumEpistasisInterpolator(P=2, seq_length=9, alphabet_type="rna")
    model.set_data(X=X_train, y=y_train)
    y_pred = model.predict()
    y_pred.loc[X_test, :].to_csv("results/mei.test.csv")

    D = DeltaPOperator(4, 9, P=2)
    e2 = quad(D, y_pred["y"]) / D.n_p_faces
    sd = np.sqrt(e2)
    print(D.n_p_faces)
    print("e2 = {:.2f}".format(e2))
    print("sd = {:.2f}".format(sd))
    print(train["y"].min(), train["y"].max())
    
    
    D = DeltaPOperator(4, 9, P=1)
    print(D.n_p_faces)
    e2 = quad(D, y_pred["y"]) / D.n_p_faces
    sd = np.sqrt(e2)
    print("e2 = {:.2f}".format(e2))
    print("sd = {:.2f}".format(sd))
    print(train["y"].min(), train["y"].max())

    print("Done")
