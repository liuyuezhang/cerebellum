import cupy as cp


def sigmoid(x):
    return 1/(1 + cp.exp(-x))
