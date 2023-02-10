import jax
import jax.numpy as jnp
from dataclasses import dataclass
import equinox as eqx
from typing import Tuple
from chex import Scalar, Array

from . import attention

def test_basic_transformer():
  key = jax.random.PRNGKey(0)
  m = attention.TransformerDecoder(
          key, 32, 4, 32, 3, 4)

  outs = m(jnp.ones(32, dtype=jnp.int32))
  outs2 = m.generate(key, jnp.ones(5, dtype=jnp.int32))
