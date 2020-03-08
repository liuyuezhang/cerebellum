import cupy as cp
from model.functions import relu_derive


# FGSM attack code
def fgsm(x, epsilon, grad):
    # reshape
    grad = grad.reshape(x.shape)
    # Collect the element-wise sign of the data gradient
    sign_grad = cp.sign(grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_x = x + epsilon * sign_grad
    # Adding clipping to maintain [0,1] range
    perturbed_x = cp.clip(perturbed_x, 0, 1)
    # Return the perturbed image
    return perturbed_x


# def calc_cerebellum_grad(model, e):
#     grad_pc = model.pc.W.T @ e
#     if model.args.granule == 'fc':
#         grad_gc = (relu_derive(model.gc.y) * model.gc.W).T
#     grad = grad_gc @ grad_pc
#     return grad


def calc_cerebellum_grad(model, e):
    grad_pc = (relu_derive(model.pc.x) * model.pc.W).T @ e
    if model.args.granule == 'fc':
        grad_gc = model.gc.W.T
    grad = grad_gc @ grad_pc
    return grad
