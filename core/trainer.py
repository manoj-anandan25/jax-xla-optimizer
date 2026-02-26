import jax
from core.model import forward

@jax.jit
def step(params, x, y):
    def loss_fn(p):
        pred = forward(p, x)
        return jax.numpy.mean((pred - y)**2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads