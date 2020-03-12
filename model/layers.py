import numpy as np
import cupy as cp
import math

import chainer
import chainer.functions as F
import model.functions as f
from chainer import Link
from chainer import initializers
from chainer import configuration


class MA(Link):
    def __init__(self, beta=0.99):
        super(MA, self).__init__()
        self.beta = beta
        self.ma = None

        self.register_persistent('beta')
        self.register_persistent('ma')

    def forward(self, x):
        if self.ma is None:
            # xp = cp.get_array_module(x)
            self.ma = cp.zeros((1, ) + x.shape[1:])
        if configuration.config.train:
            u = x.data.mean(axis=0)
            self.ma = self.beta * self.ma + (1 - self.beta) * u
        return x - self.ma


class LC(Link):
    def __init__(self, in_size, out_size, k, no_bias=False):
        super(LC, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.no_bias = no_bias

        with self.init_scope():
            start = np.random.randint(in_size - k, size=out_size)
            self.idx = np.array(f.n_ranges(start, start + k, return_flat=False))
            self.W = chainer.Parameter(
                initializers.Normal(1. / math.sqrt(in_size)),
                (out_size, k))
            if not no_bias:
                self.b = chainer.Parameter(0, (out_size,))

        self.register_persistent('no_bias')
        self.register_persistent('idx')

    def forward(self, x):
        x_idx = x[:, self.idx]
        z = self.W * x_idx
        y = F.sum(z, axis=-1)
        if not self.no_bias:
            y += self.b
        return y


class RC(Link):
    def __init__(self, in_size, out_size, k, no_bias=False):
        super(RC, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.no_bias = no_bias

        with self.init_scope():
            self.idx = np.random.randint(in_size, size=(out_size, k))
            self.W = chainer.Parameter(
                initializers.Normal(1. / math.sqrt(in_size)),
                (out_size, k))
            if not no_bias:
                self.b = chainer.Parameter(0, (out_size,))

        self.register_persistent('no_bias')
        self.register_persistent('idx')

    def forward(self, x):
        x_idx = x[:, self.idx]
        z = self.W * x_idx
        y = F.sum(z, axis=-1)
        if not self.no_bias:
            y += self.b
        return y
