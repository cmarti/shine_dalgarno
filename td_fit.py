import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from scipy.stats import pearsonr


class ThermodynamicModel(torch.nn.Module):
    def __init__(self, seq0='AGGAGGUA'):
        super().__init__()
        self.seq0 = seq0
        self.l = len(seq0)
        self.upstream_seq = 'CCG'
        self.downstream_seq = 'UGAG'
        self.npos = len(self.upstream_seq) + len(self.downstream_seq) + 9 - self.l
        self.init_params()
        
    def seqs_to_x(self, seqs):
        x = np.array([[c for c in s] for s in seqs])
        X = torch.tensor(np.stack([x == c for c in 'ACGU'], axis=2)).to(dtype=torch.float32)
        return(X)

    def encode_seqs(self, seqs):
        extended_seqs = [[(self.upstream_seq + seq + self.downstream_seq)[i:i + self.l] for seq in seqs]
                         for i in range(self.npos)]
        X = torch.stack([self.seqs_to_x(s) for s in extended_seqs], axis=3)
        return(X)
        
    def init_params(self):
        self.theta_raw = torch.nn.Parameter(torch.normal(torch.zeros((self.l, 4))))
        self.theta0 = torch.nn.Parameter(torch.zeros(1, self.npos))
        self.background = torch.nn.Parameter(torch.zeros(1,))
        self.max_value = torch.nn.Parameter(torch.ones(1,))
        self.log_sigma2 = torch.nn.Parameter(torch.zeros(1,))

    @property
    def theta(self):
        return(self.theta_raw - self.theta_raw.mean(1).unsqueeze(1))
        # return(self.theta_raw)

    @property
    def sigma2(self):
        return(torch.exp(self.log_sigma2))
    
    def summary(self, pred=None, obs=None):
        print('===========================')
        print('Log-likelihood = {:.2f}'.format(model.history[-1]))
        print('======= Parameters ========')
        for param, values in self.get_params().items():
            print('--- {} ---'.format(param))
            print(values)

        if pred is not None and obs is not None:
            r = pearsonr(pred, obs)[0]
            print('======= Predictions ========')
            print('Pearson r = {:.2f}'.format(r))        
       
    def predict(self, X):
        phi = self.theta0 + torch.tensordot(X, self.theta, dims=((1, 2), (0, 1)))
        mu = torch.exp(-phi).sum(axis=1)
        p = mu / (1 + mu)
        yhat = self.background + self.max_value * p
        return(yhat)
    
    def fit(self, seqs, y, y_var, n_iter=1000, lr=0.1):
        X = self.encode_seqs(seqs)
        y = torch.Tensor(y)
        y_var = torch.Tensor(y_var)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        pbar = tqdm(range(n_iter))
        for i in pbar:
            optimizer.zero_grad()
            yhat = self.predict(X)
            loss = torch.nn.functional.gaussian_nll_loss(yhat, y, y_var + self.sigma2, reduction='mean')
            loss.backward()
            history.append(loss.detach().item())
            optimizer.step()
            pbar.set_postfix({'loss': history[-1]})
        self.history = history
    
    def get_params(self):
        params = {
                  'theta0': self.theta0.detach().numpy()[0],
                  'theta': pd.DataFrame(self.theta.detach().numpy(), columns=['A', 'C', 'G', 'U']),
                  'background': self.background.detach().numpy()[0],
                  'max_value': self.max_value.detach().numpy()[0],
                  'sigma2': self.sigma2.detach().numpy()[0],
                  }
        return(params)    


if __name__ == "__main__":
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0)
    X_train, y_train, y_var_train = (
        train.index.values,
        train.y.values,
        train.y_var.values,
    )

    model = ThermodynamicModel(seq0='AGGAGGUA')
    model.fit(X_train, y_train, y_var_train, n_iter=2500, lr=0.01)
    
    params = model.state_dict()
    torch.save(params, 'results/thermodynamic_model.pth')
    
    history = pd.DataFrame({'loss': model.history})
    history.to_csv('results/thermodynamic_model.loss.csv')
    
    
    

