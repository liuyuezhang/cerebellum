from .utils import *


# Cerebellum
class Cerebellum:
    def __init__(self, gc, pc):
        self.gc = gc
        self.pc = pc

    def forward(self, x):
        x = self.gc.forward(x)
        x = self.pc.forward(x)
        return x

    def backward(self, e):
        self.pc.backward(e)


# Pyramid cells
class FC:
    def __init__(self, m, n, lr, update='hebbian', ltd='none'):
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)
        self.e = None

        # ltd
        self.ltd = ltd
        self.b = cp.zeros(self.in_shape)
        self.beta = 0.99

        # update method (hebbian or gradient)
        self.update = update
        self.derive = None

        self.lr = lr
        self.alpha = 0.99
        self.r_w = cp.zeros((n, m))

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape) - self.b
        if self.ltd is 'ma':
            self.b = self.beta * self.b + (1 - self.beta) * self.b
        # forward
        z = self.W @ self.x
        self.y = sigmoid(z).reshape(self.out_shape)
        # save derivative
        if self.update is 'gradient':
            self.derive = self.y * (1 - self.y)
        return self.y

    def backward(self, e):
        # input
        self.e = e.reshape(self.out_shape)
        # calc derivative
        if self.update is 'hebbian':
            yw = self.x.T
        elif self.update is 'gradient':
            yw = self.derive @ self.x.T
        # update (RMSprop)
        g_w = yw * self.e
        self.r_w = self.alpha * self.r_w + (1-self.alpha) * (g_w * g_w)
        self.W -= self.lr / cp.sqrt(1e-8 + self.r_w) * g_w


# Granule cells
class Random:
    def __init__(self, m, n):
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        z = self.W @ self.x
        self.y = sigmoid(z).reshape(self.out_shape)
        return self.y
