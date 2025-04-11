import numpy as np
import pandas as pd

import torch

from torch.nn import Parameter
from torch.nn.functional import gaussian_nll_loss
from tqdm import tqdm
from itertools import product


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.background = Parameter(0.47 * torch.ones(1))
        self.log_sigma2 = Parameter(torch.zeros(1))
        self.init_params()

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2)

    def calc_log_likelihood(self, X, y, y_var):
        yhat = self.predict(X)
        sigma2 = y_var + self.sigma2
        return -gaussian_nll_loss(yhat, y, sigma2, reduction="sum")

    def fit(self, seqs, y, y_var, n_iter=1500, lr=0.02):
        X = self.encode_seqs(seqs)
        y = torch.Tensor(y)
        y_var = torch.Tensor(y_var)
        n_obs = y.shape[0]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, maximize=True)
        history = []
        pbar = tqdm(range(n_iter))
        
        for i in pbar:
            optimizer.zero_grad()
            ll = self.calc_log_likelihood(X, y, y_var)
            ll.backward()
            history.append(ll.detach().item())
            optimizer.step()
            pbar.set_postfix({"ll": history[-1] / n_obs})
        self.history = history
    
    def save(self, fpath):
        params = self.state_dict()
        torch.save(params, fpath)
    
    def load(self, fpath):
        params = torch.load(fpath)
        self.load_state_dict(params)
    

class ThermodynamicModel(Model):
    def __init__(self, seq_length=8, alphabet='ACGU'):
        self.l = seq_length
        self.n_alleles = len(alphabet)
        self.alphabet = alphabet
        self.n = int(self.n_alleles ** self.l)
        self.upstream_seq = "CCG"
        self.downstream_seq = "UGAG"
        self.ul = len(self.upstream_seq)
        self.dl = len(self.downstream_seq)
        self.npos = self.ul + self.dl + 9 - self.l - 1
        self.seqs = ["".join(c) for c in product(alphabet, repeat=self.l)]
        self.seqs_idx = pd.Series(range(self.n), index=self.seqs)
        self.X = self.seqs_to_x(self.seqs)
        super().__init__()
        
    def seqs_to_x(self, seqs):
        x = np.array([[c for c in s] for s in seqs])
        X = torch.tensor(np.stack([x == c for c in "ACGU"], axis=2))
        return X.to(dtype=torch.float32)

    def encode_seqs(self, seqs):
        us, ds = self.upstream_seq, self.downstream_seq
        
        seqs_idx = []
        for i in range(self.npos):
            seqs_i = [(us + seq + ds)[i : i + self.l] for seq in seqs]
            seqs_idx.append(self.seqs_idx.loc[seqs_i].values)
        seqs_idx = np.array(seqs_idx).T
        
        return torch.Tensor(seqs_idx).to(dtype=torch.long)

    def init_params(self):
        theta_raw0 = torch.zeros((self.l, 4))
        theta_raw0[0, 0] = -2
        theta_raw0[1, 2] = -2
        theta_raw0[2, 2] = -2
        theta_raw0[3, 0] = -2
        theta_raw0[4, 2] = -2
        theta_raw0[5, 2] = -2
        theta_raw0[6, 3] = -2
        theta_raw0[7, 0] = -2
        self.theta_raw = torch.nn.Parameter(theta_raw0)
        self.theta0 = torch.nn.Parameter(6.0 * torch.ones(1))

    @property
    def theta(self):
        return self.theta_raw - self.theta_raw.mean(1).unsqueeze(1)

    def get_phi0(self):
        return self.theta0 + torch.einsum("ila,la->i", self.X, self.theta)

    def X_to_phi(self, X):
        phi0 = self.get_phi0()
        return phi0[X]

    def predict(self, X):
        phi = self.X_to_phi(X)
        mu = torch.logsumexp(-phi, axis=1)
        yhat = torch.logaddexp(self.background, mu)
        return yhat

    def save_phi0(self, fpath):
        phi0 = pd.DataFrame(
            {"phi": self.get_phi0().detach().numpy()}, index=self.seqs
        )
        phi0.to_csv(fpath)


class RNAModel(Model):
    def init_params(self):
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.ones(1))
    
    def encode_seqs(self, X):
        return(torch.Tensor(X))

    def predict(self, X):
        mu = -(self.alpha + self.beta * X)
        yhat = torch.logaddexp(self.background, mu)
        return yhat
