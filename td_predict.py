import pandas as pd

import torch
from itertools import product
from td_fit import ThermodynamicModel


if __name__ == "__main__":
    seqs = ["".join(c) for c in product("ACGU", repeat=9)]

    with torch.no_grad():
        model = ThermodynamicModel()
        params = torch.load("results/thermodynamic_model.pth")
        model.load_state_dict(params)

        X = model.encode_seqs(seqs)
        y_pred = model.predict(X)
        phi = model.X_to_phi(X)
        
        output = pd.DataFrame({"y_pred": y_pred}, index=seqs)
        colnames = ["dg{}".format(i + 1) for i in range(model.npos)]
        phi = pd.DataFrame(phi, columns=colnames, index=seqs)
        output = output.join(phi)
        
        output.to_csv("results/thermodynamic_model.pred.csv")
