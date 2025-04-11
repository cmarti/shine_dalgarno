import numpy as np
import torch
import pandas as pd

from itertools import product
from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    vcregression = pd.read_csv("results/vcregression.full.csv", index_col=0)
    seqdeft = pd.read_csv("results/e_coli.seqdeft_inference.csv", index_col=0)
    b_sub = pd.read_csv("results/b_sub.seqdeft_inference.csv", index_col=0)
    tdmodel = pd.read_csv("results/thermodynamic_model.pred.csv", index_col=0)
    rnamodel = pd.read_csv("results/rnamodel.pred.csv", index_col=0)

    # Calc SeqDEFT visualization
    space = SequenceSpace(X=b_sub.index.values, y=np.log(b_sub["Q_star"]))
    rw = WMWalk(space)
    rw.calc_visualization(Ns=1, n_components=20)
    rw.write_tables(
        prefix="results/b_sub.seqdeft",
        nodes_format="pq",
        edges_format="npz",
        write_edges=True,
    )

    # Calc VC regression visualization
    print("Calculating visualization for VC regression MAP")
    space = SequenceSpace(X=vcregression.index.values, y=vcregression.y.values)
    rw = WMWalk(space)
    for mean_function in [1, 1.5, 2, 2.5]:
        print("\tStationary mean function of {}".format(mean_function))
        rw.calc_visualization(mean_function=mean_function, n_components=20)
        rw.write_tables(
            prefix="results/vcregression.map.mf_{}".format(mean_function),
            nodes_format="pq",
            write_edges=False,
        )
    mean_function = 2.0

    # Calc Thermodynamic model visualization
    space = SequenceSpace(X=tdmodel.index.values, y=tdmodel["y_pred"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=mean_function, n_components=20)
    rw.write_tables(
        prefix="results/thermodynamic_model",
        nodes_format="pq",
        write_edges=False,
    )
    
    # Calc RNAmodel model visualization
    space = SequenceSpace(X=rnamodel.index.values, y=rnamodel["pred"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=2.0, n_components=20)
    rw.write_tables(
        prefix="results/rnamodel",
        nodes_format="pq",
        write_edges=False,
    )
    exit()

    # Calc thermodynamic model energies visualization
    theta = torch.load("results/thermodynamic_model_delta.pth")["theta"].numpy()
    seqs = np.array(["".join(c) for c in product("ACGU", repeat=9)])
    y = np.logaddexp(-theta, 0.47)
    print(y.mean(), y.max())
    space = SequenceSpace(X=seqs, y=y)
    space.write_edges("results/thermodynamic_model_theta.edges.npz")
    rw = WMWalk(space)
    mean_function = 2.0
    print("\tStationary mean function of {}".format(mean_function))
    rw.calc_visualization(mean_function=mean_function, n_components=20)
    rw.write_tables(
        prefix="results/thermodynamic_model_theta.mf_{}".format(mean_function),
        nodes_format="pq",
        write_edges=False,
    )
