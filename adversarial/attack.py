import cupy as cp


# FGSM attack code
def fgsm(x, epsilon, grad):
    xp = cp.get_array_module(x)
    # reshape
    grad = grad.reshape(x.shape)
    # Collect the element-wise sign of the data gradient
    sign_grad = xp.sign(grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_x = x + epsilon * sign_grad
    # Adding clipping to maintain [0,1] range
    perturbed_x = xp.clip(perturbed_x, 0, 1)
    # Return the perturbed image
    return perturbed_x
