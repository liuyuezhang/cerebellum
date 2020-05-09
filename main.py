import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
import model.functions as f

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators, optimizers, serializers
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from param import get_parser
import wandb


def train(args, epoch, train_iter, model, optimizer):
    accuracies = []
    losses = []
    last_idx = 0

    chainer.config.train = True
    train_iter.reset()  # reset (reshuffle)
    for train_batch in train_iter:
        data, label = concat_examples(train_batch, args.gpu_id)

        # forward
        output = model.forward(data)
        target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)
        loss = F.mean_squared_error(output, target)

        # backward
        model.cleargrads()
        loss.backward()

        # update
        optimizer.update()

        # accuracy
        accuracy = F.accuracy(output, label)
        accuracies.append(to_cpu(accuracy.array))
        losses.append(to_cpu(loss.array))

        # log
        batch_idx = train_iter.current_position + 1
        if batch_idx - last_idx >= args.log_interval:
            last_idx = batch_idx
            mean_loss = np.mean(losses[-args.log_interval:])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_iter.dataset),
                100. * batch_idx / len(train_iter.dataset),
                mean_loss))
            if args.wandb:
                wandb.log({"train_loss": mean_loss, "batch": (epoch - 1) * len(train_iter.dataset) + batch_idx})

    if args.wandb:
        wandb.log({"train_acc": np.mean(accuracies), "epoch": epoch})

    return np.mean(accuracies)


def test(args, epoch, test_iter, model):
    accuracies = []

    chainer.config.train = False
    test_iter.reset()  # reset
    for test_batch in test_iter:
        with chainer.no_backprop_mode():
            data, label = concat_examples(test_batch, args.gpu_id)

            # forward
            output = model.forward(data)

            # accuracy
            accuracy = F.accuracy(output, label)
            accuracies.append(to_cpu(accuracy.array))

    print('test_accuracy:{:.04f}'.format(np.mean(accuracies)))
    if args.wandb:
        wandb.log({"test_acc": np.mean(accuracies), "epoch": epoch})

    return np.mean(accuracies)


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # name
    method = args.granule
    if args.granule == 'lc' or args.granule == 'rc':
        method += ('-' + str(args.k))
    name = args.env + '_' + method + '_' + args.ltd + '_' + str(args.n_hidden) + '_' + str(args.seed)
    print(name)
    if args.wandb:
        wandb.init(name=name, project="cerebellum", entity="liuyuezhang", config=args)

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # data
    if args.env == 'gaussian':
        train_data, test_data = get_gaussian()
        in_size = 1000
        out_size = 10
    elif args.env == 'mnist':
        train_data, test_data = get_mnist(withlabel=True, ndim=1)
        in_size = 28 * 28
        out_size = 10
    elif args.env == 'cifar10':
        train_data, test_data = get_cifar10(withlabel=True, ndim=1)
        in_size = 3 * 32 * 32
        out_size = 10
    else:
        raise NotImplementedError
    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    from model.cerebellum import Cerebellum
    model = Cerebellum(in_size=in_size, out_size=out_size, args=args)

    # device
    if args.gpu_id >= 0:
        model.to_gpu(args.gpu_id)

    # optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=args.lr, alpha=0.99)
    elif args.optimizer == 'sgd':
        optimizer = optimizers.SGD(lr=args.lr)
    else:
        raise NotImplementedError
    optimizer.setup(model.purkinje)

    # train
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(1, args.epoch + 1):
        train_acc = train(args, epoch, train_iter, model, optimizer)
        test_acc = test(args, epoch, test_iter, model)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # save
            if args.save:
                serializers.save_npz(wandb.run.dir + '/model.pkl', model)

    print('best_train_accuracy:{:.04f}'.format(best_train_acc))
    print('best_test_accuracy:{:.04f}'.format(best_test_acc))
    if args.wandb:
        wandb.log({"best_train_acc": best_train_acc, "best_test_acc": best_test_acc})


if __name__ == '__main__':
    main()
