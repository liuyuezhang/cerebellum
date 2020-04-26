import cupy as cp

import chainer.functions as F
from chainer import Variable


# FGSM
def fgsm(model, data, target, eps):
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
    x = cp.clip(x, 0, 1)
    return x


# BIM / PGD attack (random_start = False / True)
def bim(model, data, target, eps, alpha=2/225, steps=40, random_start=False):
    # initialize
    if random_start:
        x = data + cp.random.uniform(-eps, +eps, data.size)
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
        x = cp.clip(x + eta, 0, 1)

    return x


# # MIM attack
# def mim(model, data, target, eps, alpha=2/225, steps=40, random_start=False):
#     # initialize
#     if random_start:
#         x = data + cp.random.uniform(-eps, +eps, data.size)
#     else:
#         x = data
#
#     for _ in range(steps):
#         # forward
#         x0 = Variable(x)
#         output = model.forward(x0, attack=True)
#
#         # gradient
#         loss = F.mean_squared_error(output, target)
#         model.cleargrads()
#         loss.backward()
#         grad = x0.grad
#
#         # attack
#         x = x0.data + alpha * cp.sign(grad)
#
#         # clip
#         eta = cp.clip(x - data, -eps, +eps)
#         x = cp.clip(x + eta, 0, 1)
#
#     return x

