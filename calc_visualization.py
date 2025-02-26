import numpy as np
import pandas as pd
from gpmap.src.randwalk import WMWalk
from gpmap.src.space import SequenceSpace

if __name__ == "__main__":
    vcregression = pd.read_csv("results/vcregression.full.csv", index_col=0)
    seqdeft = pd.read_csv("results/seqdeft.full.csv", index_col=0)
    # tdmodel = pd.read_csv("data/thermodynamic_model.csv", index_col=0)

    # # Calc VC regreesion visualization
    # print('Calculating visualization for VC regression MAP')
    # space = SequenceSpace(X=vcregression.index.values, y=vcregression.y.values)
    # rw = WMWalk(space)
    # for mean_function in [1, 1.5, 2, 2.5]:
    #     print('\tStationary mean function of {}'.format(mean_function))
    #     rw.calc_visualization(mean_function=mean_function, n_components=20)
    #     rw.write_tables(
    #         prefix="results/vcregression.map.mf_{}".format(mean_function),
    #         nodes_format="pq", write_edges=False
    #     )
    # exit()

    # # Calc SeqDEFT regreesion visualization
    # space = SequenceSpace(X=seqdeft.index.values, y=np.log(seqdeft["Q_star"]))
    # rw = WMWalk(space)
    # rw.calc_visualization(Ns=1, n_components=20)
    # rw.write_tables(
    #     prefix="data/seqdeft",
    #     nodes_format="pq",
    #     edges_format="npz",
    #     write_edges=True,
    # )

    # Calc VC regreesion visualization
    space = SequenceSpace(X=vcregression.index.values, y=vcregression.y.values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=2.5, n_components=20)
    rw.write_tables(
        prefix="results/vcregression", nodes_format="pq", write_edges=True
    )

    # # Calc thermodynamic model visualization
    # space = SequenceSpace(X=tdmodel.index.values, y=tdmodel["y_pred"].values)
    # rw = WMWalk(space)
    # rw.calc_visualization(mean_function=2.25, n_components=20)
    # rw.write_tables(
    #     prefix="data/thermodynamic_model", nodes_format="pq", write_edges=False
    # )
