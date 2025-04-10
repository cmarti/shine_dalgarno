import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from itertools import product


class ThermodynamicModel(torch.nn.Module):
    def __init__(self, seq_length=8, alphabet='ACGU'):
        super().__init__()
        self.l = seq_length
        self.n_alleles = len(alphabet)
        self.alphabet = alphabet
        self.n = int(self.n_alleles ** self.l)
        self.upstream_seq = "CCG"
        self.downstream_seq = "UGAG"
        self.npos = (
            len(self.upstream_seq) + len(self.downstream_seq) + 9 - self.l - 1
        )
        self.init_params()
        self.seqs = ["".join(c) for c in product(alphabet, repeat=self.l)]
        self.seqs_idx = pd.Series(range(self.n), index=self.seqs)
        self.X = self.seqs_to_x(self.seqs)

    def seqs_to_x(self, seqs):
        x = np.array([[c for c in s] for s in seqs])
        X = torch.tensor(np.stack([x == c for c in "ACGU"], axis=2)).to(
            dtype=torch.float32
        )
        return X

    def encode_seqs(self, seqs):
        seqs_idx = [
            self.seqs_idx.loc[
                [
                    (self.upstream_seq + seq + self.downstream_seq)[
                        i : i + self.l
                    ]
                    for seq in seqs
                ]
            ].values
            for i in range(self.npos)
        ]
        seqs_idx = np.array(seqs_idx).T
        return torch.Tensor(seqs_idx).to(dtype=torch.long)

    def init_params(self):
        theta_raw0 = torch.zeros((self.l, 4))
        # Initialize with a AGGAGGUA
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
        self.background = torch.nn.Parameter(0.47 * torch.ones(1))
        self.log_sigma2 = torch.nn.Parameter(torch.zeros(1))

    @property
    def theta(self):
        return self.theta_raw - self.theta_raw.mean(1).unsqueeze(1)

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2)

    def get_phi0(self):
        return self.theta0 + torch.einsum("ila,la->i", self.X, self.theta)

    def X_to_phi(self, X):
        phi0 = self.get_phi0()
        phi = torch.stack([phi0[X[:, i]] for i in range(self.npos)], axis=1)
        return phi

    def predict(self, X):
        phi = self.X_to_phi(X)
        mu = torch.logsumexp(-phi, axis=1)
        yhat = torch.logaddexp(self.background, mu)
        return yhat

    def calc_log_likelihood(self, X, y, y_var):
        yhat = self.predict(X)
        return -torch.nn.functional.gaussian_nll_loss(
            yhat, y, y_var + self.sigma2, reduction="sum"
        )

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

    def save_phi0(self, fpath):
        phi0 = pd.DataFrame(
            {"phi": self.get_phi0().detach().numpy()}, index=self.seqs
        )
        phi0.to_csv(fpath)


if __name__ == "__main__":
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0)
    X_train, y_train, y_var_train = (
        train.index.values,
        train.y.values,
        train.y_var.values,
    )

    model = ThermodynamicModel()
    model.fit(X_train, y_train, y_var_train, n_iter=1500, lr=0.02)

    model.save_phi0("results/thermodynamic_model_additive.csv")
    params = model.state_dict()
    torch.save(params, "results/thermodynamic_model.pth")

    history = pd.DataFrame({"ll": model.history})
    history.to_csv("results/thermodynamic_model.ll.csv")
