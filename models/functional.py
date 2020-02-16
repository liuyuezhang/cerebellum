import cupy as cp
import numpy as np


def sigmoid(x):
    return 1/(1 + cp.exp(-x))


def sigmoid_derive(x):
    return x * (1 - x)


def relu(x):
    return x * (x > 0)


def relu_derive(x):
    return x > 0


def n_ranges(start, end, return_flat=True):
    """
    Returns n ranges, n being the length of start (or end,
    they must be the same length) where each value in
    start represents the start of a range, and a value
    in end at the same index the end of it
    ----
    a: np.array
       1D array representing the start of a range.
       Each value in start must be <= than that
       of stop in the same index
    ----
    Returns:
      All ranges flattened in a 1d array if return_flat is True
      otherwise an array of arrays with a range in each
    """
    # lengths of the ranges
    lens = end - start
    # repeats starts as many times as lens
    start_rep = np.repeat(start, lens)
    # helper mask with as many True in each row
    # as value in same index in lens
    arr = np.arange(lens.max())
    m = arr < lens[:, None]
    # ranges in a flattened 1d array
    # right term is a cumcount up to each len
    ranges = start_rep + (arr * m)[m]
    # returns if True otherwise in split arrays
    if return_flat:
        return ranges
    else:
        return np.split(ranges, np.cumsum(lens)[:-1])
