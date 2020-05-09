import chainer
import chainer.functions as F
import chainer.links as L
import model.layers as l


# Cerebellum
class Cerebellum(chainer.Chain):
    def __init__(self, in_size, out_size, args):
        super(Cerebellum, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.args = args

        with self.init_scope():
            # Granule cells
            if args.granule == 'fc':
                self.granule = L.Linear(in_size, args.n_hidden, nobias=not args.bias)
            elif args.granule == 'lc':
                self.granule = l.LC(in_size, args.n_hidden, args.k, no_bias=not args.bias)
            elif args.granule == 'rc':
                self.granule = l.RC(in_size, args.n_hidden, args.k, no_bias=not args.bias)

            # Nonlinearity
            self.nonlinear = F.relu

            # Long Term Depression
            if args.ltd == 'ma':
                self.norm = l.MA(beta=args.beta)
            else:
                self.norm = None

            # Purkinje cells
            self.purkinje = L.Linear(args.n_hidden, out_size, nobias=not args.bias)

    def forward(self, x, attack=False):
        if not attack:
            with chainer.no_backprop_mode():
                z = self.project(x)
        else:
            z = self.project(x)
        y = self.purkinje.forward(z)
        return y

    def project(self, x):
        z = self.granule.forward(x)
        y = self.nonlinear(z)
        if self.norm is not None:
            y = self.norm(y)
        return y
