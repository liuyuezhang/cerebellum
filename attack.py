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


def attack(args, model, test_iter, epsilon):
    accuracies = []
    perturbed_accuracies = []

    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)
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


def main():
    # args
    name = "mnist_rand-fc_fc_10000_ma-False_hebbian-rmsprop_0"
    api = wandb.Api()
    runs = api.runs("liuyuezhang/cerebellum")
    for run in runs:
        if run.name == name:
            id = run.id
            args = Bunch(run.config)
    print(run.name)

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data
    if args.env == 'gaussian':
        test_data = get_gaussian()[1]
    elif args.env == 'mnist':
        test_data = get_mnist(withlabel=True, ndim=1)[1]
    elif args.env == 'cifar10':
        test_data = get_cifar10(withlabel=True, ndim=1)[1]
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    wandb.restore('model.pkl', run_path="liuyuezhang/cerebellum/" + id)
    model = load()

    # attack
    eps = 0.3
    attack(args, model, test_iter, eps)


if __name__ == '__main__':
    main()
