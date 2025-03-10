import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpmap.src.randwalk import WMWalk
from gpmap.src.space import SequenceSpace
from gpmap.src.inference import VCregression
from time import time
import gpmap.src.plot.ds as dplot
import gpmap.src.plot.mpl as mplot


if __name__ == "__main__":
    results = []
    for alphabet, max_length in [("dna", 12), ("protein", 5)]:
        for seq_length in range(1, max_length + 1):
            lambdas = np.geomspace(0.001, 10, seq_length + 1)[::-1]
            model = VCregression(
                seq_length=seq_length, alphabet_type=alphabet, lambdas=lambdas
            )
            y = model.sample()

            space = SequenceSpace(X=model.genotypes, y=y)
            rw = WMWalk(space)
            rw.calc_visualization(mean_function_perc=90, n_components=3)
            nodes = rw.nodes_df
            edges = space.get_edges_df()

            for _ in range(3):
                if y.shape[0] < 5e6:
                    t0 = time()
                    fig, axes = plt.subplots(1, 1)
                    mplot.plot_visualization(axes, nodes, edges_df=edges)
                    fig.savefig("figures/test_rendering_times.png", dpi=300)
                    plt.close(fig)
                    t1 = time() - t0
                else:
                    t1 = None

                t0 = time()
                dsg = dplot.plot_visualization(nodes, edges_df=edges)
                fig = dplot.dsg_to_fig(dsg)
                fig.savefig("figures/test_rendering_times.png", dpi=300)
                plt.close(fig)
                t2 = time() - t0

                results.append(
                    {
                        "seq_length": seq_length,
                        "type": alphabet,
                        "mpl": t1,
                        "ds": t2,
                    }
                )
                print(results[-1])

    results = pd.DataFrame(results)
    results.to_csv("results/rendering_times.csv")
