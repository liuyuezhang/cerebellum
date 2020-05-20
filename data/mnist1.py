import numpy as np
from chainer.datasets import get_mnist
from chainer.datasets import TupleDataset


def get_mnist1():
    _, data = get_mnist(withlabel=True, ndim=1)

    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    idx = (y == 0) | (y == 1)
    x = x[idx]
    y = y[idx] * 2 - 1
    data = TupleDataset(x, y)
    return data
