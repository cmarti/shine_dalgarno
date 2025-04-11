import pandas as pd

from scripts.utils import load_experimental_data
from scripts.models.models import ThermodynamicModel

if __name__ == "__main__":
    X, y, y_var = load_experimental_data()[:3]

    print('Estimating model parameters')
    model = ThermodynamicModel()
    model.fit(X, y, y_var, n_iter=1500, lr=0.02)

    print('Saving model parameters')
    model.save("results/thermodynamic_model.pth")

    print('Saving optimization history')
    history = pd.DataFrame({"ll": model.history})
    history.to_csv("results/thermodynamic_model.ll.csv")
