import pandas as pd

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    mean_function = 2.0
    data = pd.read_csv("results/rna_model.pred.csv", index_col=0)
    X, y = data.index.values, data.y_pred.values

    # Calc Thermodynamic model visualization
    space = SequenceSpace(X, y)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=mean_function, n_components=20)
    rw.write_tables(
        prefix="results/rna_model",
        nodes_format="pq",
        write_edges=False,
    )
