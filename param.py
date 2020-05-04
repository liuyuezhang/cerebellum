import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mnist', choices=('mnist', 'cifar10'))
    parser.add_argument('--attack', type=str, default='fgsm', choices=('random', 'fgsm', 'pgd'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--granule', type=str, default='fc', choices=('fc', 'lc', 'rc'),
                        help='fully, locally or randomly connected without training.')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--golgi', type=str, default='none', choices=('none', 'inhibit'))
    parser.add_argument('--purkinje', type=str, default='fc')
    parser.add_argument('--n-hidden', type=int, default=5000)
    parser.add_argument('--ltd', type=str, default='none', choices=('none', 'ma'))
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--optimizer', default='rmsprop', choices=('sgd', 'rmsprop'))
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--res-dir', type=str, default='./wandb')
    parser.add_argument('--log-adv-num', type=int, default=10)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--save', default=False, action='store_true')
    return parser
