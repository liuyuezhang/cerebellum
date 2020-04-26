import cupy as cp

import chainer.functions as F
from chainer import Variable


# FGSM
def fgsm(model, data, target, epsilon):
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
    x = x0.data + epsilon * cp.sign(grad)

    # clip
    x = cp.clip(x, 0, 1)
    return x


# BIM attack
def bim(model, data, target, epsilon, steps=20):
    # initialize
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
        x = x0.data + epsilon * cp.sign(grad)

    # clip
    x = cp.clip(x, 0, 1)
    return x

