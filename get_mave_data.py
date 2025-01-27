import pandas as pd
import numpy as np

if __name__ == "__main__":
    samples = ["dmsC.1", "dmsC.2", "dmsC.3"]
    data = pd.read_csv(
        "data/dmsc.csv", index_col=0, usecols=["Genotype"] + samples
    )
    data["n_measured"] = np.sum(~np.isnan(data[samples]), axis=1)
    data = data.loc[data["n_measured"] >= 1, :]
    data["y"] = np.nanmean(data[samples], axis=1)

    df = pd.melt(
        data.loc[data["n_measured"] > 1, samples + ["y"]], id_vars=["y"]
    ).dropna()
    sample_var = np.sum((df["value"] - df["y"]) ** 2) / (
        df.shape[0] - data.shape[0]
    )
    data["y_var"] = sample_var / data["n_measured"]
    data.to_csv("data/dmsc_processed.csv")

    # Split into test and training sets
    np.random.seed(0)
    u = np.random.uniform(size=data.shape[0])
    train = data.loc[u < 0.999, :]
    test = data.loc[u > 0.999, :]

    train.to_csv("data/dmsc.train.csv")
    test.to_csv("data/dmsc.test.csv")

    print(
        "Split into {} sequences for training and {} for testing".format(
            train.shape[0], test.shape[0]
        )
    )
