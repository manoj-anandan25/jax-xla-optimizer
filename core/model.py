import jax
import jax.numpy as jnp

def init_params(in_dim, out_dim, key):
    return {
        'w': jax.random.normal(key, (in_dim, out_dim)) * 0.02,
        'b': jnp.zeros((out_dim,))
    }

def forward(params, x):
    return jnp.dot(x, params['w']) + params['b']