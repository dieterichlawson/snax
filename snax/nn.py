import jax
import jax.numpy as jnp
from jax._src.random import KeyArray as PRNGKey
from chex import Array
from typing import Tuple, Sequence, List
from dataclasses import dataclass
from .utils import register_pytree
from .base import Layer, ParamBase

from jax.nn.initializers import glorot_normal, zeros

def Identity() -> Layer[None]:
  """A layer which passes the input through unchanged."""

  class Identity:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, None]:
      del key
      return input_dim, None

    @staticmethod
    def apply(params: None, inputs: Array) -> Array:
      del params
      return inputs

  return Identity

@register_pytree
@dataclass
class LinearParams(ParamBase):
  W: Array

def Linear(out_dim: int,
           W_init=glorot_normal()) -> Layer[LinearParams]:
  """A Linear layer (no bias)."""

  class Linear:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, LinearParams]:
      W = W_init(key, (input_dim, out_dim))
      return out_dim, LinearParams(W=W)

    @staticmethod
    def apply(params: LinearParams, inputs: Array) -> Array:
      return jnp.dot(inputs, params.W)

  return Linear

@register_pytree
@dataclass
class AffineParams(ParamBase):
  W: Array
  b: Array

def Affine(out_dim: int,
           W_init=glorot_normal(),
           b_init=zeros) -> Layer[AffineParams]:
  """An affine layer."""

  class Affine:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, AffineParams]:
      k1, k2 = jax.random.split(key)
      W = W_init(k1, (input_dim, out_dim))
      b = b_init(k2, (out_dim,))
      return out_dim, AffineParams(W=W, b=b)

    @staticmethod
    def apply(params: AffineParams, inputs: Array) -> Array:
      return jnp.dot(inputs, params.W) + params.b

  return Affine

def Dense(out_dim,
          W_init=glorot_normal(),
          b_init=zeros,
          activation=jax.nn.relu) -> Layer[AffineParams]:
  """A single-layer MLP (Affine layer with an activation)."""
  affine = Affine(out_dim, W_init=W_init, b_init=b_init)

  class Dense:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, AffineParams]:
      return affine.init(key, input_dim)

    @staticmethod
    def apply(params: AffineParams, inputs: Array) -> Array:
      return activation(affine.apply(params, inputs))

  return Dense

@register_pytree
@dataclass
class MLPParams(ParamBase):
  layer_params: Sequence[AffineParams]

def MLP(layer_dims: List[int],
        W_init=glorot_normal(),
        b_init=zeros,
        activation=jax.nn.relu,
        activate_final=False) -> Layer[MLPParams]:
  """A multi-layered perceptron."""

  layers = []
  for dim in layer_dims[:-1]:
    layers.append(Dense(dim, W_init=W_init, b_init=b_init,
                        activation=activation))
  if activate_final:
    layers.append(Dense(layer_dims[-1], W_init=W_init, b_init=b_init,
                        activation=activation))
  else:
    layers.append(Affine(layer_dims[-1], W_init=W_init, b_init=b_init))

  class MLP:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, MLPParams]:
      keys = jax.random.split(key, num=len(layer_dims))
      input_dims = [input_dim] + layer_dims[:-1]
      params = []
      for layer, key, in_dim in zip(layers, keys, input_dims):
        params.append(layer.init(key, in_dim)[1])
      return layer_dims[-1], MLPParams(params)

    @staticmethod
    def apply(params: MLPParams, inputs: Array) -> Array:
      for layer, param in zip(layers, params.layer_params):
        inputs = layer.apply(param, inputs)
      return inputs

  return MLP
