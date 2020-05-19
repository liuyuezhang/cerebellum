import numpy as np
from chainer.datasets import TupleDataset


def get_gaussian(d=500, n=1000, c=1, mu=0, sigma=1, seed=0):
    np.random.seed(seed)
    x = np.array(np.random.randn(n, d) * sigma + mu, dtype=np.float32)
    # single class: +1 / -1
    if c == 1:
        y = np.random.randint(0, 2, n) * 2 - 1
    # multiple classes
    else:
        y = np.random.randint(0, c, n)
    data = TupleDataset(x, y)
    return data
