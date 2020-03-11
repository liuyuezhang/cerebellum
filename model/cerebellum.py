from model import layers as L
from model import functions as F


# Cerebellum
class Cerebellum:
    def __init__(self, input_dim, output_dim, args):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args

        # Nonlinearity
        self.nonlinear = F.relu

        # Granule cells
        if args.granule == 'fc':
            self.granule = L.FixFC(m=input_dim, n=args.n_hidden)
        elif args.granule == 'lc':
            self.granule = L.FixLC(m=input_dim, n=args.n_hidden, k=args.k)
        elif args.granule == 'rand':
            self.granule = L.FixRand(m=input_dim, n=args.n_hidden, k=args.k)

        # Golgi cells
        if args.golgi:
            if args.granule == 'fc':
                self.golgi = L.FixFC(m=input_dim, n=args.n_hidden)
            elif args.granule == 'lc':
                self.golgi = L.FixLC(m=input_dim, n=args.n_hidden, k=args.k)
            elif args.granule == 'rand':
                self.golgi = L.FixRand(m=input_dim, n=args.n_hidden, k=args.k)

        # Purkinje cells
        if args.purkinje == 'fc':
            self.purkinje = L.FC(m=args.n_hidden, n=output_dim,
                                 ltd=args.ltd, beta=args.beta, bias=args.bias,
                                 optimization=args.optimization, lr=args.lr, alpha=args.alpha)

    def forward(self, x):
        if self.args.golgi:
            x = self.granule.forward(x) - self.golgi.forward(x)
        else:
            x = self.granule.forward(x)
        x = self.nonlinear(x)
        x = self.purkinje.forward(x)
        return x

    def backward(self, e):
        self.purkinje.backward(e)

    def train(self):
        self.purkinje.train = True

    def test(self):
        self.purkinje.train = False
