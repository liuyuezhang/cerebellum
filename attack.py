import numpy as np
import cupy as cp

import chainer
import model.functions as f

from data.gaussian import get_gaussian
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
    adv_exs = []

    chainer.config.train = False
    test_iter.reset()  # reset
    for test_batch in test_iter:
        data, label = concat_examples(test_batch, args.gpu_id)

        # Forward the test data
        output = model.forward(data)
        if args.env == 'gaussian1':
            target = cp.array(label.reshape(output.shape), dtype=output.dtype)
            pred = int(np.sign(output.item()))
        else:
            target = f.one_hot(label, out_size=output.shape[-1], dtype=output.dtype)
            pred = output.data.argmax()

        # Attack
        clip = False if args.env.startswith('gaussian') else True
        if args.attack == 'random':
            adv_data = random(data, eps, clip=clip)
        elif args.attack == 'fgsm':
            adv_data = fgsm(model, data, target, eps, clip=clip)
        elif args.attack == 'pgd':
            adv_data = pgd(model, data, target, eps, alpha=0.01, steps=40, random_start=True, clip=clip)
        else:
            raise NotImplementedError

        # Forward the test data
        adv_output = model.forward(adv_data)
        if args.env == 'gaussian1':
            adv_pred = int(np.sign(adv_output.item()))
        else:
            adv_pred = adv_output.data.argmax()

        # Calculate the accuracy
        if adv_pred == label:
            correct += 1
        else:
            # Save some adv examples for visualization later
            if args.save_img and len(adv_exs) < args.log_adv_num:
                if args.env == 'mnist':
                    img = cp.asnumpy(adv_data.reshape(28, 28))
                elif args.env == 'cifar10':
                    img = cp.asnumpy(adv_data.reshape(3, 32, 32).swapaxes(0, 2))
                adv_exs.append((img, pred, adv_pred, label))

    acc = correct / len(test_iter.dataset)
    print('eps:{:.02f} perturbed_acc:{:.04f}'.format(eps, acc))
    if args.wandb:
        wandb.log({"attack_acc": acc, "eps": eps})
        if args.save_img:
            wandb.log({"eps=" + str(eps): [wandb.Image(img, caption='pred:' + str(pred)
                                                                    + ', adv:' + str(adv_pred)
                                                                    + ', label:' + str(label))
                                           for img, pred, adv_pred, label in adv_exs]}, commit=False)

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
    if config.env == 'gaussian1':
        data = get_gaussian(d=500, n=1000, c=1, mu=0.5, sigma=1, seed=args.seed)
        test_data = data
        in_size = 500
        out_size = 1
    elif config.env == 'gaussian2':
        data = get_gaussian(d=500, n=1000, c=2, mu=0.5, sigma=1, seed=args.seed)
        test_data = data
        in_size = 500
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
    if args.env.startswith('gaussian'):
        eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    elif args.env == 'mnist':
        eps_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    elif args.env == 'cifar10':
        eps_list = [0, 2/255, 4/255, 6/255, 8/255]
    else:
        raise NotImplementedError
    for eps in eps_list:
        acc = test(args, eps, test_iter, model)

    print('max_attack_acc:{:.04f}'.format(acc))
    if args.wandb:
        wandb.log({"max_attack_acc": acc})


if __name__ == '__main__':
    main()
