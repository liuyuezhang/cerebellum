import cupy as cp

import chainer
import chainer.functions as F
import model.functions as f
from chainer import Variable

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators, serializers
from chainer.dataset import concat_examples

from adversarial.attack import fgsm
import wandb
import os
import argparse


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def test(args, epsilon, test_iter, model):
    correct = 0
    adv_exs = []

    chainer.config.train = False
    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)

        # Require grads
        data = Variable(data)

        # Forward the test data
        output = model.forward(data, attack=True)
        target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)
        pred = output.data.argmax()

        # Calculate the gradient
        loss = F.mean_squared_error(output, target)
        model.cleargrads()
        loss.backward()
        adv_data = fgsm(data.data, epsilon, data.grad)

        # Forward the test data
        adv_output = model.forward(adv_data)
        adv_pred = adv_output.data.argmax()

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

    parser.add_argument('--granule', type=str, default='fc', choices=('fc', 'lc', 'rc'),
                        help='fully, locally or randomly connected without training.')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--golgi', default=False, action='store_true')
    parser.add_argument('--purkinje', type=str, default='fc')
    parser.add_argument('--n-hidden', type=int, default=5000)
    parser.add_argument('--ltd', type=str, default='none', choices=('none', 'ma'))
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--res-dir', type=str, default='./wandb')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--log-num', type=int, default=10)
    args = parser.parse_args()

    # name
    method = args.granule
    if args.granule == 'lc' or args.granule == 'rand':
        method += ('-' + str(args.k))
    if args.golgi:
        method += '-inhibit'
    name = args.env + '_' + method + '_' + args.ltd + '_' + str(args.n_hidden) + '-' + str(args.lr) + '_' + str(
        args.seed)
    print(name)

    # find run
    api = wandb.Api()
    runs = api.runs("liuyuezhang/cerebellum")
    for run in runs:
        if run.name == name:
            run_id = run.id
            config = Bunch(run.config)
            print(run.name)
            break

    # data
    if config.env == 'gaussian':
        test_data = get_gaussian()[1]
        in_size = 1000
        out_size = 10
    elif config.env == 'mnist':
        test_data = get_mnist(withlabel=True, ndim=1)[1]
        in_size = 28 * 28
        out_size = 10
    elif config.env == 'cifar10':
        test_data = get_cifar10(withlabel=True, ndim=1)[1]
        in_size = 3 * 32 * 32
        out_size = 10
    else:
        raise NotImplementedError
    test_iter = iterators.MultiprocessIterator(test_data, 1, repeat=False, shuffle=True)

    # model
    from model.cerebellum import Cerebellum
    model = Cerebellum(in_size=in_size, out_size=out_size, args=config)
    if args.gpu_id >= 0:
        model.to_gpu(args.gpu_id)

    # load
    for file in os.listdir(args.res_dir):
        if run_id in file:
            serializers.load_npz(args.res_dir + '/' + file + '/model.pkl', model)
            break

    # attack and log
    if args.wandb:
        wandb.init(name=args.attack + '_' + name, project="cerebellum", entity="liuyuezhang")
    # eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    eps_list = [0]
    for eps in eps_list:
        test(args, eps, test_iter, model)


if __name__ == '__main__':
    main()
