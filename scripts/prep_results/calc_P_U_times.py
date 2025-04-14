import numpy as np
import pandas as pd
import tracemalloc
from time import time
from gpmap.matrix import kron
from gpmap.linop import VjProjectionOperator


def calc_Pj(n_alleles, seq_length, j):
    P0 = np.full((n_alleles, n_alleles), fill_value=1.0 / n_alleles)
    P1 = np.eye(n_alleles) - P0
    Ps = [P0, P1]
    Ps = [Ps[int(i in j)] for i in range(seq_length)]
    return kron(Ps)


if __name__ == "__main__":
    operators = {
        "Dense matrix": calc_Pj,
        "Linear Operator": VjProjectionOperator,
    }
    results = []
    for alphabet, n_alleles, max_length in [
        ("dna", 4, 12),
        ("protein", 20, 5),
    ]:
        print("Computing matrix-vector products in {} space".format(alphabet))
        for seq_length in range(2, max_length + 1):
            print("\tSequence length = {}".format(seq_length))
            n = int(n_alleles**seq_length)
            positions = np.arange(seq_length)

            for label, operator in operators.items():
                if n > 4e4 and label == "Dense matrix":
                    continue

                for i in range(20):
                    j = positions[np.random.uniform(size=seq_length) < 0.5]

                    tracemalloc.start()
                    current1, peak1 = tracemalloc.get_traced_memory()
                    t0 = time()
                    Pj = operator(n_alleles, seq_length, j)
                    t1 = time() - t0
                    current2, peak2 = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    v = np.random.normal(size=n)
                    t0 = time()
                    u = Pj @ v
                    t2 = time() - t0
                    results.append(
                        {
                            "seq_length": seq_length,
                            "n_alleles": n_alleles,
                            "type": alphabet,
                            "operator": label,
                            "overhead_time": t1,
                            "matvec_time": t2,
                            "current_memory": (current2 - current1) / 1e6,
                            "peak_memory": (peak2 - peak1) / 1e6,
                        }
                    )
                    # print(results[-1])

    results = pd.DataFrame(results)
    results.to_csv("results/times_P_U_operator.csv")
