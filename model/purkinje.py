import cupy as cp


# Purkinje cells
class FC:
    def __init__(self, m, n, ltd='none', beta=0.99, bias=False,
                 optimization='rmsprop', lr=1e-4, alpha=0.99, dropout=0.0):
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
        self.p = dropout
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
        # dropout
        if self.train:
            r = cp.random.binomial(n=1, p=self.p, size=self.in_shape)
            x = x * r
        else:
            x = x * self.p
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
