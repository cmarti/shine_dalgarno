import pandas as pd
import torch

from rnamodel_fit import RNAModel


if __name__ == "__main__":
    print("Loading computed binding energies between SD:aSD")
    energies = pd.read_csv("processed/SDaSD.energies.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0).join(energies)
    X_train, y_train, y_var_train = (
        train.dG.values,
        train.y.values,
        train.y_var.values,
    )

    print("Loading model parameters")
    model = RNAModel()
    params = torch.load("results/rnamodel.pth")
    model.load_state_dict(params)

    print("Computing predictions for the genotype-phenotype map")
    with torch.no_grad():
        energies["pred"] = model.predict(torch.Tensor(energies["dG"].values))
    energies.to_csv("results/rnamodel.pred.csv")
