import pandas as pd

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    print("Loading VC regression MAP")
    data = pd.read_csv("results/vcregression.map.csv", index_col=0)
    X, y = data.index.values, data.f.values

    print("Calculating visualization for VC regression MAP")
    space = SequenceSpace(X, y)
    rw = WMWalk(space)
    for mean_function in [1, 1.5, 2, 2.5]:
        print("\tStationary mean function of {}".format(mean_function))
        rw.calc_visualization(mean_function=mean_function, n_components=20)
        rw.write_tables(
            prefix="results/vcregression.map.mf_{}".format(mean_function),
            nodes_format="pq",
            write_edges=False,
        )
