import numpy as np
import pandas as pd

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    for species in ["b_sub", "e_coli"]:
        print("Loading SeqDEFT results from {}".format(species))
        fpath = "results/{}.seqdeft.map.csv".format(species)
        data = pd.read_csv(fpath, index_col=0)
        X, y = data.index.values, np.log(data.Q_star.values)

        print("Calculating visualization for SeqDEFT from {}".format(species))
        space = SequenceSpace(X, y)
        rw = WMWalk(space)
        rw.calc_visualization(Ns=1, n_components=20)
        rw.write_tables(
            prefix="results/{}.seqdeft.map".format(species),
            nodes_format="pq",
            edges_format="npz",
            write_edges=False,
        )
