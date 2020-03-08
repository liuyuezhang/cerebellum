from model import functions as F
import cupy as cp
import numpy as np


# Granule cells
class FC:
    def __init__(self, m, n):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))

        # nonlinearity
        self.nonlinear = F.relu

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        z = self.W @ self.x
        y = self.nonlinear(z)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y


class LC:
    def __init__(self, m, n, k=4):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)
        self.k = k

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical
        start = np.random.randint(m - k, size=n)
        self.idx = np.array(F.n_ranges(start, start + k, return_flat=False))
        stdv = 1. / cp.sqrt(k)
        self.W = cp.random.uniform(-stdv, stdv, (n, k))

        # nonlinearity
        self.nonlinear = F.relu

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        x_idx = self.x.squeeze(axis=1)[self.idx]
        z = cp.sum(self.W * x_idx, axis=1, keepdims=True)
        y = self.nonlinear(z)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y


class Rand:
    def __init__(self, m, n, k=4):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)
        self.k = k

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical
        self.idx = np.random.randint(m, size=(n, k))
        stdv = 1. / cp.sqrt(k)
        self.W = cp.random.uniform(-stdv, stdv, (n, k))

        # nonlinearity
        self.nonlinear = F.relu

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        x_idx = self.x.squeeze(axis=1)[self.idx]
        z = cp.sum(self.W * x_idx, axis=1, keepdims=True)
        y = self.nonlinear(z)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y
