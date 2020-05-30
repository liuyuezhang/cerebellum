import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
import model.functions as f

from chainer.datasets import get_mnist, get_cifar10
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.transforms import resize_contain, random_crop, random_flip
from chainer import iterators, optimizers, serializers
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from param import get_parser
import wandb


def tranform(data):
    img, label = data
    img = resize_contain(img, size=(40, 40), fill=0)
    img = random_crop(img, size=(32, 32))
    img = random_flip(img, x_random=True, y_random=False)
    return img, label


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
        if args.env.endswith('1'):
            target = cp.array(label.reshape(output.shape), dtype=output.dtype)
        else:
            target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)
        loss = F.mean_squared_error(output, target)

        # backward
        model.cleargrads()
        loss.backward()

        # update
        optimizer.update()

        # accuracy
        if args.env.endswith('1'):
            pred = int(np.sign(output.item()))
            accuracy = 1 if pred == label.item() else 0
            accuracies.append(accuracy)
        else:
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
            if args.env.endswith('1'):
                pred = int(np.sign(output.item()))
                accuracy = 1 if pred == label.item() else 0
                accuracies.append(accuracy)
            else:
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

    # data
    if args.env == 'mnist':
        train_data, test_data = get_mnist(withlabel=True, ndim=1)
        in_size = 28 * 28
        out_size = 10
    elif args.env == 'cifar10':
        train_data, test_data = get_cifar10(withlabel=True, ndim=3)
        train_data = TransformDataset(train_data, ('img', 'label'), tranform)
        in_size = 2048
        out_size = 10
    else:
        raise NotImplementedError

    # seed
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    train_iter = iterators.MultiprocessIterator(train_data, args.batch_size, repeat=False, shuffle=True)
    test_iter = iterators.MultiprocessIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    # model
    from model.cerebellum import Cerebellum
    from model.visual_cerebellum import VisualCerebellum
    if args.env == 'mnist':
        model = Cerebellum(in_size=in_size, out_size=out_size, args=args)
    elif args.env == 'cifar10':
        model = VisualCerebellum(in_size=in_size, out_size=out_size, args=args)
        model.init_visual('./pretrain/resnet50_cifar10.pkl')
    else:
        raise NotImplementedError

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
    if args.env == 'mnist':
        optimizer.setup(model.purkinje)
    elif args.env == 'cifar10':
        optimizer.setup(model.cerebellum.purkinje)
    else:
        raise NotImplementedError

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
            # early stopping
            if best_test_acc >= 0.995:
                break

    print('best_train_accuracy:{:.04f}'.format(best_train_acc))
    print('best_test_accuracy:{:.04f}'.format(best_test_acc))
    if args.wandb:
        wandb.log({"best_train_acc": best_train_acc, "best_test_acc": best_test_acc})


if __name__ == '__main__':
    main()
