from models import functional as F
import cupy as cp
import numpy as np
import pickle


# Cerebellum
class Cerebellum:
    def __init__(self, input_dim, output_dim, args):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args

        # Granule cells
        if args.granule_cell == 'rand':
            if args.granule_connect == 'fc':
                self.gc = RandFC(m=input_dim, n=args.n_hidden,)
            elif args.granule_connect == 'lc':
                self.gc = RandLC(m=input_dim, n=args.n_hidden, p=args.p)

        # Purkinje cells
        if args.purkinje_cell == 'fc':
            self.pc = FC(m=args.n_hidden, n=output_dim,
                         ltd=args.ltd, beta=args.beta, bias=args.bias, softmax=args.softmax,
                         optimization=args.optimization, lr=args.lr, alpha=args.alpha)

    def forward(self, x):
        x = self.gc.forward(x)
        x = self.pc.forward(x)
        return x

    def backward(self, e):
        self.pc.backward(e)

    def save(self, dir):
        with open(dir + '/model.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# Purkinje cells
class FC:
    def __init__(self, m, n, ltd='none', beta=0.99, bias=False, softmax=False,
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

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))
        self.bias = bias
        if self.bias:
            self.b = cp.random.uniform(-stdv, stdv, self.out_shape)

        # nonlinearity
        self.sofxmax = softmax

        # learning (hebbian or gradient)
        self.learning = 'hebbian'

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
            self.ma = self.beta * self.ma + (1 - self.beta) * x
            self.x = x - self.ma
        elif self.ltd == 'none':
            self.x = x
        # forward
        z = self.W @ self.x
        if self.bias:
            z += self.b
        if self.sofxmax:
            y = F.softmax(z)
        else:
            y = z
        # output
        self.y = y.reshape(self.out_shape)
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


# Granule cells
class RandFC:
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


class RandLC:
    def __init__(self, m, n, p=4):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)
        self.p = p

        # interface
        start = np.random.randint(m - p, size=n)
        self.idx = np.array(F.n_ranges(start, start + p, return_flat=False))
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical
        stdv = 1. / cp.sqrt(p)
        self.W = cp.random.uniform(-stdv, stdv, (n, p))

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
