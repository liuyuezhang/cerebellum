import cupy as cp
import numpy as np
import model.functions as F


# Granule cells and Golgi cells
class FixFC:
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

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        y = self.W @ self.x
        # output
        self.y = y.reshape(self.out_shape)
        return self.y


class FixLC:
    def __init__(self, m, n, k=4):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)
        self.k = k

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical

        stdv = 1. / cp.sqrt(k)
        self.W = cp.random.uniform(-stdv, stdv, (n, k))

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        x_idx = self.x.squeeze(axis=1)[self.idx]
        y = cp.sum(self.W * x_idx, axis=1, keepdims=True)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y


class FixRand:
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

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        x_idx = self.x.squeeze(axis=1)[self.idx]
        y = cp.sum(self.W * x_idx, axis=1, keepdims=True)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y


# Purkinje cells
class FC:
    def __init__(self, m, n, ltd='none', beta=0.99, bias=False,
                 optimization='rmsprop', lr=1e-4, alpha=0.99):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)
        self.e = cp.zeros(self.out_shape)

        # ltd
        self.ltd = ltd
        if self.ltd == 'ma':
            self.ma = cp.zeros(self.in_shape)
            self.beta = beta

        # dropout
        self.train = False

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))
        self.bias = bias
        if self.bias:
            self.b = cp.random.uniform(-stdv, stdv, self.out_shape)

        # optimization
        self.optimization = optimization
        self.lr = lr
        if self.optimization == 'rmsprop':
            self.alpha = alpha
            self.r_w = cp.zeros((n, m))
            if self.bias:
                self.r_b = cp.zeros(self.out_shape)

    def forward(self, x):
        # input
        x = x.reshape(self.in_shape)
        # ltd
        if self.ltd == 'ma':
            if self.train:
                self.ma = self.beta * self.ma + (1 - self.beta) * x
            self.x = x - self.ma
        elif self.ltd == 'none':
            self.x = x
        # forward
        z = self.W @ self.x
        if self.bias:
            z += self.b
        # output
        self.y = z.reshape(self.out_shape)
        return self.y

    def backward(self, e):
        # input
        self.e = e.reshape(self.out_shape)
        # learning
        yw = self.x.T
        if self.bias:
            yb = cp.ones(self.out_shape)
        # optimization
        g_w = yw * self.e
        if self.bias:
            g_b = yb * self.e
        if self.optimization == 'sgd':
            self.W -= self.lr * g_w
            if self.bias:
                self.b -= self.lr * g_b
        elif self.optimization == 'rmsprop':
            self.r_w = self.alpha * self.r_w + (1-self.alpha) * (g_w * g_w)
            if self.bias:
                self.r_b = self.alpha * self.r_b + (1-self.alpha) * (g_b * g_b)
            self.W -= self.lr / cp.sqrt(1e-8 + self.r_w) * g_w
            if self.bias:
                self.b -= self.lr / cp.sqrt(1e-8 + self.r_b) * g_b
