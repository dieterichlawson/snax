from . import utils
import jax.numpy as jnp

def test_flip_first_n():
  x = jnp.array(range(10))
  out1 = utils.flip_first_n(x, 3)
  assert jnp.allclose(out1[:3], jnp.array([2, 1, 0]))
  out2 = utils.flip_first_n(x, 6)
  assert jnp.allclose(out2[:6], jnp.array([5, 4, 3, 2, 1, 0]))
  out3 = utils.flip_first_n(x, 10)
  assert jnp.allclose(out3, jnp.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
  _ = utils.flip_first_n(x, 0)
