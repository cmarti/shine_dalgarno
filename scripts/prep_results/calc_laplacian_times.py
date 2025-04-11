import sys
import numpy as np
import pandas as pd
import tracemalloc
from time import time
from gpmap.matrix import calc_Kn_matrix, calc_cartesian_product
from gpmap.linop import LaplacianOperator


def calc_L(n_alleles, seq_length):
    Kn = calc_Kn_matrix(k=n_alleles)
    L = calc_cartesian_product([Kn] * seq_length)
    d = -L.sum(1).A1.flatten()
    L.setdiag(d)
    return -L


if __name__ == "__main__":
    operators = {'Sparse matrix': calc_L,
                 'Linear Operator': LaplacianOperator}
    results = []
    for alphabet, n_alleles, max_length in [
        ("dna", 4, 12),
        ("protein", 20, 5),
    ]:
        for seq_length in range(1, max_length + 1):
            n = int(n_alleles**seq_length)
            
            for label, operator in operators.items():
                if n > 5e6 and label == 'Sparse matrix':
                    continue
                
                for i in range(20):
                    
                    tracemalloc.start()
                    current1, peak1 = tracemalloc.get_traced_memory()
                    t0 = time()
                    L = operator(n_alleles, seq_length)
                    t1 = time() - t0
                    current2, peak2 = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    v = np.random.normal(size=n)
                    t0 = time()
                    u = L @ v
                    t2 = time() - t0
                    results.append(
                        {
                            "seq_length": seq_length,
                            'n_alleles': n_alleles,
                            "type": alphabet,
                            "operator": label,
                            "overhead_time": t1,
                            "matvec_time": t2,
                            "current_memory": (current2 - current1) / 1e6,
                            "peak_memory": (peak2 - peak1) / 1e6,
                        }
                    )
                    print(results[-1])
                
    results = pd.DataFrame(results)
    results.to_csv("results/laplacian_times.csv")
