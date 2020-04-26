import cupy as cp

import chainer
import model.functions as f

from data.gaussian import get_gaussian
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators, serializers
from chainer.dataset import concat_examples

from adversarial.attack import fgsm, bim
import wandb
import os
from param import get_parser


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

        # Forward the test data
        output = model.forward(data)
        target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)
        pred = output.data.argmax()

        # Attack
        if args.attack == 'fgsm':
            adv_data = fgsm(model, data, target, epsilon)
        elif args.attack == 'bim':
            adv_data = bim(model, data, target, epsilon, steps=20)
        else:
            raise NotImplementedError

        # Forward the test data
        adv_output = model.forward(adv_data)
        adv_pred = adv_output.data.argmax()

        # Calculate the accuracy
        if adv_pred == label:
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_exs) < args.log_adv_num:
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

    return acc


def main():
    # args
    parser = get_parser()
    args = parser.parse_args()

    # name
    method = args.granule
    if args.granule == 'lc' or args.granule == 'rc':
        method += ('-' + str(args.k) + '-' + str(args.golgi))
    name = args.env + '_' + method + '_' + args.ltd + '_' + str(args.n_hidden) + '_' + str(args.seed)
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

    # load
    for file in os.listdir(args.res_dir):
        if run_id in file:
            serializers.load_npz(args.res_dir + '/' + file + '/model.pkl', model)
            break

    # device
    if args.gpu_id >= 0:
        model.to_gpu(args.gpu_id)

    # attack and log
    if args.wandb:
        wandb.init(name=args.attack + '-' + name, project="cerebellum", entity="liuyuezhang")
    eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for eps in eps_list:
        acc = test(args, eps, test_iter, model)

    print('max_attack_acc:{:.04f}'.format(acc))
    if args.wandb:
        wandb.log({"max_attack_acc": acc})


if __name__ == '__main__':
    main()
