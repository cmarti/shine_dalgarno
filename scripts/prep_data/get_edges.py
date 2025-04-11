import numpy as np

from itertools import product
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    X = np.array(["".join(c) for c in product("ACGU", repeat=9)])
    y = np.random.normal(size=X.shape[0])
    space = SequenceSpace(X, y)
    space.write_edges('results/edges.npz')
    