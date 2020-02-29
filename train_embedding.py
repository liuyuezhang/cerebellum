import argparse

import numpy as np
import cupy as cp
import chainer.functions as F

import chainer
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
from chainer import optimizers, serializers


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='embedding')
    parser.add_argument('--env', type=str, default='cifar10', choices=('mnist', 'cifar10'))
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=0.0)

    parser.add_argument('--gpu-id', type=int, default=0, help='cpu: -1')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--log-interval', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data
    if args.env == 'mnist':
        train_data, test_data = get_mnist(withlabel=True, ndim=3)
        shape = (1, 28, 28)
    elif args.env == 'cifar10':
        train_data, test_data = get_cifar10(withlabel=True, ndim=3)
        shape = (3, 32, 32)
    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)

    # model
    from embedding.ae import AE
    model = AE(size=shape[-2:], in_channels=shape[0]).to_gpu(args.gpu_id)

    # optimizer
    optimizer = optimizers.RMSprop(lr=1e-4)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay))

    # train
    for epoch in range(1, args.epoch + 1):
        losses = []
        train_iter.reset()  # reset (reshuffle)
        for train_batch in train_iter:
            # data
            data, label = concat_examples(train_batch, args.gpu_id)

            # model
            output = model(data)
            loss = F.mean_squared_error(output, data)

            # Calculate the gradients in the network
            model.cleargrads()
            loss.backward()

            # Update all the trainable parameters
            optimizer.update()

            # log
            losses.append(to_cpu(loss.array))
            batch_idx = train_iter.current_position
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_iter.dataset),
                    100. * batch_idx / len(train_iter.dataset), np.mean(losses)))

    # save
    serializers.save_npz('./res/' + args.env + '.model', model)


if __name__ == '__main__':
    main()
