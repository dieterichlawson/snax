import jax
import jax.numpy as jnp
from collections import namedtuple

import typing
from typing import NamedTuple, List

from .base import Module

from jax.nn.initializers import glorot_normal, normal, ones, zeros

def Identity():
  """A layer which passes the input through unchanged."""

  def init(unused_key, input_shape):
    return input_shape, None

  def apply(params, inputs):
    return inputs

  return Module(init, apply)

class LinearParams(NamedTuple):

  W: jnp.ndarray

def Linear(out_dim, W_init=glorot_normal()):
  """A Linear layer (no bias)."""

  def init(key, input_dim):
    W = W_init(key, (input_dim, out_dim))
    return [out_dim], LinearParams(W)

  def apply(params, inputs):
    return jnp.dot(inputs, params.W)

  return Module(init, apply)

class AffineParams(NamedTuple):

  W: jnp.ndarray
  b: jnp.ndarray

def Affine(out_dim, W_init=glorot_normal(), b_init=normal()):
  """An affine layer."""

  def init(key, input_dim):
    k1, k2 = jax.random.split(key)
    W, b = W_init(k1, (input_dim, out_dim)), b_init(k2, (out_dim,))
    return out_dim, AffineParams(W, b)

  def apply(params, inputs):
    return jnp.dot(inputs, params.W) + params.b

  return Module(init, apply)

def Dense(out_dim,
          W_init=glorot_normal(),
          b_init=normal(),
          activation=jax.nn.relu):
  """A single-layer MLP (Affine layer with an activation)."""
  affine = Affine(out_dim, W_init=W_init, b_init=b_init)

  def init(key, input_dim):
    return affine.init(key, input_dim)

  def apply(params, inputs):
    return activation(affine.apply(params, inputs))

  return Module(init, apply)

class MLPParams(NamedTuple):

  layer_params: List[AffineParams]

def MLP(layer_dims,
        W_init=glorot_normal(),
        b_init=normal(),
        activation=jax.nn.relu,
        activate_final=False):
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

  def init(key, input_dim):
    keys = jax.random.split(key, num=len(layer_dims))
    input_dims = [input_dim] + layer_dims[:-1]
    params = []
    for layer, key, in_dim in zip(layers, keys, input_dims):
      params.append(layer.init(key, in_dim)[1])
    return layer_dims[-1], MLPParams(params)

  def apply(params, inputs):
    for layer, param in zip(layers, params.layer_params):
      inputs = layer.apply(param, inputs)
    return inputs

  return Module(init, apply)

def Residual(layer):

  def init(key, input_dim):
    out_dim, params = layer.init(key, input_dim)
    assert input_dim == out_dim, "Residual layers must have the same input and output dims."
    return out_dim, params

  def apply(params, inputs):
    return inputs + layer.apply(params, inputs)

  return Module(init, apply)
