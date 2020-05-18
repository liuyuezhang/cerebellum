import numpy as np
from chainer.datasets import TupleDataset


def get_gaussian(d=500, n=1000, mu=0, sigma=1, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, d) * sigma + mu
    y = np.random.randint(0, 2, n) * 2 - 1
    data = TupleDataset(x, y)
    return data
