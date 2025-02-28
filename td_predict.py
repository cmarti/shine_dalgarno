import pandas as pd

import torch
from itertools import product
from td_fit import ThermodynamicModel


if __name__ == "__main__":
    seqs = [''.join(c) for c in product('ACGU', repeat=9)]
    
    with torch.no_grad():
        model = ThermodynamicModel(seq0='AGGAGGUA')
        params = torch.load('results/thermodynamic_model.pth', weights_only=True)
        model.load_state_dict(params)
        
        X = model.encode_seqs(seqs)
        y_pred = model.predict(X)
        output = pd.DataFrame({'y_pred': y_pred}, index=seqs)
        
        phi = pd.DataFrame((model.theta0 + torch.tensordot(X, model.theta, dims=((1, 2), (0, 1)))).numpy(), 
                            columns=['dg{}'.format(i+1) for i in range(8)], index=seqs)
        output = output.join(phi)
        output.to_csv('results/thermodynamic_model.pred.csv')
