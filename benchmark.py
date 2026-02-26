import jax
import jax.numpy as jnp
import time
from core.model import init_params
from core.trainer import step

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (128, 128))
y = jax.random.normal(key, (128, 10))
params = init_params(128, 10, key)

# Warmup (Compilation)
_ = step(params, x, y)

# Measure
t0 = time.time()
for _ in range(50):
    l, g = step(params, x, y)
t1 = time.time()

print(f"JIT time per step (ms): {(t1-t0)/50*1000:.4f}")