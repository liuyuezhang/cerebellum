import model.granule as granule
import model.purkinje as purkinje


# Cerebellum
class Cerebellum:
    def __init__(self, input_dim, output_dim, args):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args

        # Granule cells
        if args.granule == 'fc':
            self.gc = granule.FC(m=input_dim, n=args.n_hidden,)
        elif args.granule == 'lc':
            self.gc = granule.LC(m=input_dim, n=args.n_hidden, p=args.p)
        elif args.granule == 'rand':
            self.gc = granule.Rand(m=input_dim, n=args.n_hidden, p=args.p)

        # Purkinje cells
        if args.purkinje == 'fc':
            self.pc = purkinje.FC(m=args.n_hidden, n=output_dim,
                                  ltd=args.ltd, beta=args.beta, bias=args.bias, softmax=args.softmax,
                                  optimization=args.optimization, lr=args.lr, alpha=args.alpha)

    def forward(self, x):
        x = self.gc.forward(x)
        x = self.pc.forward(x)
        return x

    def backward(self, e):
        self.pc.backward(e)
