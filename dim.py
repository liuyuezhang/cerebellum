import argparse
import wandb
from utils import *

import numpy as np
import cupy as cp
import chainer
import chainer.functions as F

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
from chainer import serializers


def train(args, model, embedding, train_iter, epoch):
    train_losses = []
    train_accuracies = []

    train_iter.reset()  # reset (reshuffle)
    for train_batch in train_iter:
        data, label = concat_examples(train_batch, args.gpu_id)
        target = cp.zeros((10, 1))
        target[label] = 1

        # model
        output = model.forward(data)
        error = output - target
        loss = 0.5 * cp.mean(error ** 2)
        model.backward(error)

        # Calculate the loss
        train_losses.append(to_cpu(loss))

        # Calculate the accuracy
        accuracy = F.accuracy(output.reshape(1, -1), label)
        train_accuracies.append(to_cpu(accuracy.array))

        # log
        batch_idx = train_iter.current_position + 1
        if batch_idx % args.log_interval == 0:
            mean_loss = np.mean(train_losses[-args.log_interval:])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_iter.dataset),
                100. * batch_idx / len(train_iter.dataset),
                mean_loss))
            if args.wandb:
                wandb.log({"train_loss": mean_loss, "batch": (epoch - 1) * len(train_iter.dataset) + batch_idx})

    if args.wandb:
        wandb.log({"train_acc": np.mean(train_accuracies), "epoch": epoch})


def test(args, model, embedding, test_iter, epoch):
    test_losses = []
    test_accuracies = []

    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)
        target = cp.zeros((10, 1))
        target[label] = 1

        # # embedding
        # if args.embedding:
        #     with chainer.no_backprop_mode():
        #         data = embedding.embed(data)

        # Forward the test data
        output = model.forward(data)
        error = output - target
        loss = 0.5 * cp.mean(error ** 2)

        # Calculate the loss
        test_losses.append(to_cpu(loss))

        # Calculate the accuracy
        accuracy = F.accuracy(output.reshape(1, -1), label)
        test_accuracies.append(to_cpu(accuracy.array))

    print('test_loss:{:.04f} test_accuracy:{:.04f}'.format(
        np.mean(test_losses), np.mean(test_accuracies)))
    if args.wandb:
        wandb.log({"test_acc": np.mean(test_accuracies), "epoch": epoch})


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='cerebellum')
    parser.add_argument('--env', type=str, default='mnist', choices=('mnist', 'cifar10'))
    parser.add_argument('--batch-size', type=int, default=1)  # todo: batch size
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
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

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--log-interval', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data and embbeding
    if args.env == 'gaussian':
        train_data, test_data = get_gaussian()
        input_dim = 1000
        output_dim = 10
    elif args.env == 'mnist':
        train_data, test_data = get_mnist(withlabel=True, ndim=1)
        embedding = None
        input_dim = 28 * 28
        output_dim = 10
    elif args.env == 'cifar10':
        if args.embedding:
            train_data, test_data = get_cifar10(withlabel=True, ndim=3)
            from embedding.ae import AE
            embedding = AE(size=(32, 32), in_channels=3).to_gpu(args.gpu_id)
            serializers.load_npz('./res/' + args.env + '.model', embedding)
            input_dim = 8 * 15 * 15
            output_dim = 10
        else:
            train_data, test_data = get_cifar10(withlabel=True, ndim=1)
            embedding = None
            input_dim = 3 * 32 * 32
            output_dim = 10
    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    from model.cerebellum import Cerebellum
    model = Cerebellum(input_dim=input_dim, output_dim=output_dim, args=args)

    # train
    test(args, model, embedding, test_iter, 0)
    for epoch in range(1, args.epoch + 1):
        train(args, model, embedding, train_iter, epoch)
        test(args, model, embedding, test_iter, epoch)


if __name__ == '__main__':
    main()
