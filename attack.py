import numpy as np
import cupy as cp

import chainer
import model.functions as f

from data.mini_mnist import get_mini_mnist
from chainer.datasets import get_mnist, get_cifar10
from chainer import iterators, serializers
from chainer.dataset import concat_examples

from adversarial.attack import random, fgsm, pgd
import wandb
import os
from param import get_parser


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def test(args, eps, test_iter, model):
    correct = 0
    datas = []
    outputs = []
    adv_datas = []
    adv_outputs = []
    grad_infos = []
    labels = []

    chainer.config.train = False
    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)

        # Forward the test data
        output = model.forward(data)
        if args.env.endswith('1'):
            target = cp.array(label.reshape(output.shape), dtype=output.dtype)
        else:
            target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)

        # Attack
        if args.attack == 'random':
            adv_data = random(data, eps, clip=True)
        elif args.attack == 'fgsm':
            adv_data, grad_info = fgsm(model, data, target, eps, clip=True)
        elif args.attack == 'pgd':
            adv_data, grad_info = pgd(model, data, target, eps, alpha=0.01, steps=40, random_start=True, clip=True)
        else:
            raise NotImplementedError

        # Forward the test data
        adv_output = model.forward(adv_data)
        adv_pred = adv_output.data.argmax()

        # Calculate the accuracy
        if adv_pred == label:
            correct += 1
        if args.save:
            datas.append(cp.asnumpy(data))
            outputs.append(cp.asnumpy(output.array))
            adv_datas.append(cp.asnumpy(adv_data))
            adv_outputs.append(cp.asnumpy(adv_output.array))
            grad_infos.append(cp.asnumpy(grad_info))
            labels.append(cp.asnumpy(label))

    acc = correct / len(test_iter.dataset)
    print('eps:{:.02f} perturbed_acc:{:.04f}'.format(eps, acc))
    if args.wandb:
        wandb.log({"attack_acc": acc, "eps": eps})
    if args.save:
        np.savez(wandb.run.dir + '/data_' + str(eps), data=np.array(datas), output=np.array(outputs),
                 adv_data=np.array(adv_datas), adv_output=np.array(adv_outputs),
                 grad_info=np.array(grad_infos), label=np.array(labels))

    return acc


def main():
    # args
    parser = get_parser()
    args = parser.parse_args()

    # name
    method = args.granule
    if args.granule == 'lc' or args.granule == 'rc':
        method += ('-' + str(args.k))
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
    if config.env == 'mnist1':
        test_data = get_mini_mnist(c=1)
        in_size = 28 * 28
        out_size = 1
    elif config.env == 'mnist2':
        test_data = get_mini_mnist(c=2)
        in_size = 28 * 28
        out_size = 2
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

    # seed
    np.random.seed(config.seed)
    cp.random.seed(config.seed)

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
        wandb.init(name=args.attack + '-' + name, project="cerebellum", entity="liuyuezhang", config=args)
    if args.env.startswith('mnist'):
        eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    elif args.env == 'cifar10':
        eps_list = [2/255, 8/255]
    else:
        raise NotImplementedError
    for eps in eps_list:
        acc = test(args, eps, test_iter, model)

    print('max_attack_acc:{:.04f}'.format(acc))
    if args.wandb:
        wandb.log({"max_attack_acc": acc})


if __name__ == '__main__':
    main()
