import pandas as pd

import torch
from itertools import product
from scripts.models.models import ThermodynamicModel


if __name__ == "__main__":
    seqs = ["".join(c) for c in product("ACGU", repeat=9)]

    with torch.no_grad():
        print('Loading inferred thermodynamic model')
        model = ThermodynamicModel()
        model.load("results/thermodynamic_model.pth")

        print("Computing predictions for the genotype-phenotype map")
        X = model.encode_seqs(seqs)
        y_pred = model.predict(X)
        
        print("Computing binding energies at each register")
        phi = model.X_to_phi(X)
        
        print("Storing results")
        output = pd.DataFrame({"y_pred": y_pred}, index=seqs)
        colnames = ["dg{}".format(i + 1) for i in range(model.npos)]
        phi = pd.DataFrame(phi, columns=colnames, index=seqs)
        output = output.join(phi)
        
        output.to_csv("results/thermodynamic_model.pred.csv")
