import pandas as pd
from models import RNAModel


if __name__ == "__main__":
    print("Loading computed binding energies between SD:aSD")
    energies = pd.read_csv("processed/rna_model.energies.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0).join(energies)
    X, y, y_var = train.dG.values, train.y.values, train.y_var.values

    print("Fitting calibration model")
    model = RNAModel()
    model.fit(X, y, y_var, n_iter=1500, lr=0.02)

    print("Storing model parameters")
    model.save("results/rna_model.pth")
