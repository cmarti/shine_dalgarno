import pandas as pd
import torch

from tqdm import tqdm


class RNAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.background = torch.nn.Parameter(0.47 * torch.ones(1))
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.log_sigma2 = torch.nn.Parameter(torch.zeros(1))

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2)

    def predict(self, X):
        mu = -(self.alpha + self.beta * X)
        yhat = torch.logaddexp(self.background, mu)
        return yhat

    def calc_log_likelihood(self, X, y, y_var):
        yhat = self.predict(X)
        return -torch.nn.functional.gaussian_nll_loss(
            yhat, y, y_var + self.sigma2, reduction="sum"
        )

    def fit(self, X, y, y_var, n_iter=1000, lr=0.1):
        X = torch.Tensor(X)
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


if __name__ == "__main__":
    print("Loading computed binding energies between SD:aSD")
    energies = pd.read_csv("processed/SDaSD.energies.csv", index_col=0)
    train = pd.read_csv("processed/dmsc.train.csv", index_col=0).join(energies)
    X_train, y_train, y_var_train = (
        train.dG.values,
        train.y.values,
        train.y_var.values,
    )

    print("Fitting calibration model")
    model = RNAModel()
    model.fit(X_train, y_train, y_var_train, n_iter=1500, lr=0.02)

    print("Storing model parameters")
    params = model.state_dict()
    torch.save(params, "results/rnamodel.pth")
