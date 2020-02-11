from .functional import *
import pickle


# Cerebellum
class Cerebellum:
    def __init__(self, input_dim, output_dim, args):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args

        # Granule cells
        if args.granule_cell == 'randfc':
            self.gc = RandFC(m=input_dim, n=args.n_hidden, bias=args.bias, nonlinearity=args.nonlinearity)

        # Purkinje cells
        if args.purkinje_cell == 'fc':
            self.pc = FC(m=args.n_hidden, n=output_dim, ltd=args.ltd, beta=args.beta, bias=args.bias,
                         nonlinearity=args.nonlinearity,
                         learning=args.learning, optimization=args.optimization, lr=args.lr, alpha=args.alpha)

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
    def __init__(self, m, n, ltd='none', beta=0.99, bias=False, nonlinearity='sigmoid',
                 learning='hebbian', optimization='rmsprop', lr=1e-4, alpha=0.99):
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
        self.nonlinearity = nonlinearity
        if self.nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_derive = sigmoid_derive
        elif self.nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_derive = relu_derive

        # learning (hebbian or gradient)
        self.learning = learning
        if self.learning == 'gradient':
            self.derive = cp.zeros(self.out_shape)

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
        z = self.W @ self.x + self.b if self.bias else self.W @ self.x
        y = self.nonlinear(z)
        # output
        self.y = y.reshape(self.out_shape)
        # save derivative
        if self.learning == 'gradient':
            self.derive = self.nonlinear_derive(self.y)
        return self.y

    def backward(self, e):
        # input
        self.e = e.reshape(self.out_shape)
        # learning
        if self.learning == 'hebbian':
            yw = self.x.T
            if self.bias:
                yb = cp.ones(self.out_shape)
        elif self.learning == 'gradient':
            yw = self.derive @ self.x.T
            if self.bias:
                yb = self.derive
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
    def __init__(self, m, n, bias=False, nonlinearity='sigmoid'):
        # shape
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # interface
        self.x = cp.zeros(self.in_shape)
        self.y = cp.zeros(self.out_shape)

        # initialization is critical
        stdv = 1. / cp.sqrt(m)
        self.W = cp.random.uniform(-stdv, stdv, (n, m))
        self.bias = bias
        if self.bias:
            self.b = cp.random.uniform(-stdv, stdv, self.out_shape)

        # nonlinearity
        self.nonlinearity = nonlinearity
        if self.nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_derive = sigmoid_derive
        elif self.nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_derive = relu_derive

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        z = self.W @ self.x + self.b if self.bias else self.W @ self.x
        y = self.nonlinear(z)
        # output
        self.y = y.reshape(self.out_shape)
        return self.y
