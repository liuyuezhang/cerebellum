import cupy as cp
import pickle


def sigmoid(x):
    return 1/(1 + cp.exp(-x))


def sigmoid_derive(x):
    return x * (1 - x)


def relu(x):
    return x * (x > 0)


def relu_derive(x):
    return x > 0


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
