import wandb
import os
import argparse
from utils import *

import cupy as cp

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators
from chainer.dataset import concat_examples

from adversarial.fgsm import calc_cerebellum_grad, fgsm


def test(args, model, test_iter, epsilon):
    correct = 0
    adv_exs = []

    model.test()
    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, 0)
        target = cp.zeros((10, 1))
        target[label] = 1

        # Forward the test data
        output = model.forward(data)
        pred = output.argmax()
        error = output - target

        # Calculate the gradient
        grad = calc_cerebellum_grad(model, error)
        adv_data = fgsm(data, epsilon, grad)

        # Forward the test data
        adv_output = model.forward(adv_data)
        adv_pred = adv_output.argmax()

        # Calculate the accuracy
        if adv_pred == label:
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_exs) < args.log_num:
                img = cp.asnumpy(adv_data.reshape(28, 28))
                adv_exs.append((img, pred, adv_pred, label))

    acc = correct / len(test_iter.dataset)
    print('eps:{:.02f} perturbed_acc:{:.04f}'.format(epsilon, acc))
    if args.wandb:
        wandb.log({"attack_acc": acc, "eps": epsilon})
        wandb.log({"eps=" + str(epsilon): [wandb.Image(img,
                                                       caption='pred:' + str(pred) + ', adv:' + str(
                                                           adv_pred) + ', label:' + str(label))
                                           for img, pred, adv_pred, label in adv_exs]}, commit=False)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='fgsm', choices=('fgsm', 'bim', 'mim'))
    parser.add_argument('--env', type=str, default='mnist', choices=('mnist', 'cifar10'))
    parser.add_argument('--seed', type=int, default=0)

    # parser.add_argument('--embedding', default=False, actionn='store_true')
    parser.add_argument('--granule', type=str, default='fc', choices=('fc', 'lc', 'rand'),
                        help='fully, locally or randomly random connected without training.')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--purkinje', type=str, default='fc')
    parser.add_argument('--n-hidden', type=int, default=5000)
    parser.add_argument('--ltd', type=str, default='none', choices=('none', 'ma'))
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--optimization', type=str, default='rmsprop', choices=('sgd', 'rmsprop'))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=0.0)

    parser.add_argument('--res-dir', type=str, default='./wandb')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--log-num', type=int, default=10)
    args = parser.parse_args()

    # name
    # # embedding
    # embed = 'embed' if args.embedding else 'none'
    # granule cell
    granule = args.granule
    if args.granule == 'lc' or args.granule == 'rand':
        granule += ('-' + str(args.k))
    # purkinje cell
    purkinje = args.purkinje
    # bias
    bias = args.ltd + '-' + str(args.bias)

    # learning
    learning = args.optimization + '-' + str(args.weight_decay)
    name = args.env + '_' + granule + '_' + purkinje + '_' \
           + str(args.n_hidden) + '-' + str(args.lr) + '_' + bias + '_' + learning + '_' + str(args.seed)
    print("test name: " + name)

    # find run
    api = wandb.Api()
    runs = api.runs("liuyuezhang/cerebellum")
    for run in runs:
        if run.name == name:
            id = run.id
            config = Bunch(run.config)
            print("run name: " + run.name)
            break

    # data
    if config.env == 'gaussian':
        test_data = get_gaussian()[1]
    elif config.env == 'mnist':
        test_data = get_mnist(withlabel=True, ndim=1)[1]
    elif config.env == 'cifar10':
        test_data = get_cifar10(withlabel=True, ndim=1)[1]
    test_iter = iterators.MultiprocessIterator(test_data, 1, repeat=False, shuffle=True)

    # model
    for file in os.listdir(args.res_dir):
        if id in file:
            model = load(args.res_dir + '/' + file + '/model.pkl')
            break

    # attack and log
    if args.wandb:
        wandb.init(name=args.attack + '_' + name, project="cerebellum", entity="liuyuezhang")
    eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for eps in eps_list:
        test(args, model, test_iter, eps)


if __name__ == '__main__':
    main()
