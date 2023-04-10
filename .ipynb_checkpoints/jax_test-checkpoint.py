import jax.numpy as jnp
import jax
import numpy as np

def fn(x):
    return x + x*x + x*x*x

# x = np.random.randn(10000, 10000).astype(dtype='float32')
# jax_fn = jax.jit(fn)
# x = jnp.array(x)
# jax_fn(x)

fn_first = jax.grad(fn)
fn_second = jax.grad(jax.grad(fn))
fn_third = jax.grad(jax.grad(jax.grad(fn)))
print("First order: {}".format(fn_first(1.0)))
print("Second order: {}".format(fn_second(1.0)))
print("Third order: {}".format(fn_third(1.0)))