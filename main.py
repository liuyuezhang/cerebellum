import argparse
import wandb

import numpy as np
import cupy as cp

import chainer.functions as F

from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu


def train(args, model, train_iter, epoch):
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

        # log
        loss = to_cpu(loss)
        batch_idx = train_iter.current_position + 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_iter.dataset),
                100. * batch_idx / len(train_iter.dataset), loss))
            if args.wandb:
                wandb.log({"train_loss": loss, "batch": (epoch - 1) * len(train_iter.dataset) + batch_idx})


def test(args, model, test_iter, epoch):
    test_losses = []
    test_accuracies = []

    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)
        target = cp.zeros((10, 1))
        target[label] = 1

        # Forward the test data
        output = model.forward(data)
        error = output - target
        loss = 0.5 * cp.mean(error ** 2)

        # Calculate the loss
        test_losses.append(to_cpu(loss))

        # Calculate the accuracy
        accuracy = F.accuracy(output.reshape(1, -1), label)
        test_accuracies.append(to_cpu(accuracy.array))

    print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
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

    parser.add_argument('--granule-cell', type=str, default='rand', choices=('rand', ))
    parser.add_argument('--granule-connect', type=str, default='fc', choices=('fc', 'lc'))
    parser.add_argument('--p', type=int, default=4)
    parser.add_argument('--purkinje-cell', type=str, default='fc')
    parser.add_argument('--n-hidden', type=int, default=1000)
    parser.add_argument('--ltd', type=str, default='none', choices=('none', 'ma'))
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--softmax', default=False, action='store_true')
    parser.add_argument('--learning', type=str, default='hebbian', choices=('hebbian', 'gradient'))
    parser.add_argument('--optimization', type=str, default='rmsprop', choices=('sgd', 'rmsprop'))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.99)

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--log-interval', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    # logger
    # granule cell
    granule_cell = args.granule_cell + '-' + args.granule_connect
    if args.granule_connect == 'lc':
        granule_cell += ('-' + str(args.p))
    # purkinje cell
    purkinje_cell = args.purkinje_cell
    if args.softmax:
        purkinje_cell += '-softmax'
    # bias
    bias = args.ltd + '-' + str(args.bias)
    # learning
    learning = args.learning + '-' + args.optimization
    name = args.env + '_' + granule_cell + '_' + purkinje_cell + '_' \
           + str(args.n_hidden) + '_' + bias + '_' + learning + '_' + str(args.seed)
    print(name)
    if args.wandb:
        wandb.init(name=name, project="cerebellum", entity="liuyuezhang", config=args)

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data
    if args.env == 'mnist':
        train_data, test_data = get_mnist(withlabel=True, ndim=1)
        input_dim = 28 * 28
        output_dim = 10
    elif args.env == 'cifar10':
        train_data, test_data = get_cifar10(withlabel=True, ndim=1)
        input_dim = 32 * 32 * 3
        output_dim = 10
    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    from models.cerebellum import Cerebellum
    model = Cerebellum(input_dim=input_dim, output_dim=output_dim, args=args)

    # train
    test(args, model, test_iter, 0)
    for epoch in range(1, args.epoch + 1):
        train(args, model, train_iter, epoch)
        test(args, model, test_iter, epoch)

    # save
    model.save(dir=wandb.run.dir)


if __name__ == '__main__':
    main()
