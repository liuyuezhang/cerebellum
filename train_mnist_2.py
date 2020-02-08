import argparse
import wandb

import numpy as np
import cupy as cp

import chainer.functions as F

from chainer.datasets import mnist
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
                wandb.log({"train_loss": loss, "batch": (epoch-1) * len(train_iter.dataset) + batch_idx})


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
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--granule-cell', type=str, default='random', choices=('random', 'pca', 'ica'))
    parser.add_argument('--purkinje-cell', type=str, default='fc', choices=('fc', 'lc'))
    parser.add_argument('--update', type=str, default='hebbian', choices=('hebbian', 'gradient'))
    parser.add_argument('--ltd', type=str, default='none', choices=('none', ))

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--wandb', type=int, default=True)
    parser.add_argument('--log-interval', type=int, default=1000)

    args = parser.parse_args()

    # logger
    name = 'mnist' + '_' + args.granule_cell + '-' + args.purkinje_cell + \
           '-' + args.update + '-' + args.ltd + '_' + str(args.seed)
    if args.wandb:
        wandb.init(name=name, project="cerebellum", entity="liuyuezhang")

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data
    train_data, test_data = mnist.get_mnist(withlabel=True, ndim=1)
    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    lr = args.lr
    from models.cerebellum import Cerebellum, FC, Random
    gc = Random(m=28 * 28, n=1000)
    pc = FC(m=1000, n=10, lr=lr, update=args.update, ltd=args.ltd)
    model = Cerebellum(gc=gc, pc=pc)

    # train
    for epoch in range(1, args.epoch + 1):
        train(args, model, train_iter, epoch)
        test(args, model, test_iter, epoch)


if __name__ == '__main__':
    main()
