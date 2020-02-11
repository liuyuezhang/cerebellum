import cupy as cp


def sigmoid(x):
    return 1/(1 + cp.exp(-x))


def sigmoid_derive(x):
    return x * (1 - x)


def relu(x):
    return x * (x > 0)


def relu_derive(x):
    return x > 0
