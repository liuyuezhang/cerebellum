import cupy as cp

import chainer.functions as F
from chainer import Variable


# Random noise
def random(data, eps, clip=True):
    # attack
    x = data + cp.random.uniform(-eps, +eps, data.shape, dtype=cp.float32)
    # clip
    if clip:
        x = cp.clip(x, 0, 1)
    return x


# FGSM
def fgsm(model, data, target, eps, clip=True):
    # initialize
    x = data

    # forward
    x0 = Variable(x)
    output = model.forward(x0, attack=True)

    # gradient
    loss = F.mean_squared_error(output, target)
    model.cleargrads()
    loss.backward()
    grad = x0.grad

    # attack
    x = x0.data + eps * cp.sign(grad)

    # clip
    if clip:
        x = cp.clip(x, 0, 1)
    return x


# BIM / PGD attack (random_start = False / True)
def pgd(model, data, target, eps, alpha=0.01, steps=40, random_start=True, clip=True):
    # initialize
    if random_start:
        x = data + cp.random.uniform(-eps, +eps, data.shape, dtype=cp.float32)
    else:
        x = data

    for _ in range(steps):
        # forward
        x0 = Variable(x)
        output = model.forward(x0, attack=True)

        # gradient
        loss = F.mean_squared_error(output, target)
        model.cleargrads()
        loss.backward()
        grad = x0.grad

        # attack
        x = x0.data + alpha * cp.sign(grad)

        # clip
        eta = cp.clip(x - data, -eps, +eps)
        x = data + eta
        if clip:
            x = cp.clip(x, 0, 1)

    return x
