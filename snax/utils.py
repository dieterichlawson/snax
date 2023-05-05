import jax
import jax.numpy as jnp
from chex import Array, ArrayTree
from typing import Iterator

def flip_first_n(x: Array, n: int) -> Array:
  xlen = x.shape[0]
  inds = jnp.arange(xlen - 1, stop=-1, step=-1)
  inds -= (xlen - n)
  return x[inds]

def identity_init(key, shape, dtype=jnp.float32):
  assert len(shape) == 2
  return jnp.eye(shape[0], M=shape[1], dtype=dtype)

def stack_identity_init(num_reps):

  def init(key, shape, dtype=jnp.float32):
    assert len(shape) == 2
    assert shape[1] % num_reps == 0
    ms = []
    for _ in range(num_reps):
      ms.append(jnp.eye(shape[0], M=shape[1]//num_reps, dtype=dtype))
    return jnp.concatenate(ms, axis=1)

  return init
