import wandb
from utils import *

import numpy as np
import cupy as cp

import chainer.functions as F

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from adversarial.fgsm import calc_cerebellum_grad, fgsm


def test(model, test_iter, epsilon):
    accuracies = []
    perturbed_accuracies = []

    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, 0)
        target = cp.zeros((10, 1))
        target[label] = 1

        # Forward the test data
        output = model.forward(data)

        # Calculate the accuracy
        accuracy = F.accuracy(output.reshape(1, -1), label)
        accuracies.append(to_cpu(accuracy.array))

        # Calculate the gradient
        error = output - target
        grad = calc_cerebellum_grad(model, error)
        perturbed_data = fgsm(data, epsilon, grad)

        # Forward the test data
        perturbed_output = model.forward(perturbed_data)

        # Calculate the accuracy
        perturbed_accuracy = F.accuracy(perturbed_output.reshape(1, -1), label)
        perturbed_accuracies.append(to_cpu(perturbed_accuracy.array))

    print('eps:{:.02f} test_acc:{:.04f} perturbed_acc:{:.04f}'.format(
        epsilon, np.mean(accuracies), np.mean(perturbed_accuracies)))
    wandb.log({"acc": np.mean(perturbed_accuracies), "eps": epsilon})


def main():
    # args
    attack = "fgsm"  # 'fgsm', 'bim', 'mim'
    name = "mnist_fc_fc_20000_none-False_rmsprop_0"
    api = wandb.Api()
    runs = api.runs("liuyuezhang/cerebellum")
    for run in runs:
        if run.name == name:
            id = run.id
            args = Bunch(run.config)
            print(run.name)

    # data
    if args.env == 'gaussian':
        test_data = get_gaussian()[1]
    elif args.env == 'mnist':
        test_data = get_mnist(withlabel=True, ndim=1)[1]
    elif args.env == 'cifar10':
        test_data = get_cifar10(withlabel=True, ndim=1)[1]
    test_iter = iterators.MultiprocessIterator(test_data, 1, repeat=False, shuffle=False)

    # model todo: you could actually inqury with id in the ./wandb
    # wandb.restore('model.pkl', run_path="liuyuezhang/cerebellum/" + id)
    model = load('./model.pkl')

    # attack and log
    wandb.init(name=attack + '_' + name, project="cerebellum", entity="liuyuezhang")
    eps_list = [0.3]
    for eps in eps_list:
        test(args, model, test_iter, eps)


if __name__ == '__main__':
    main()
