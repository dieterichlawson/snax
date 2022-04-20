import jax
import jax.numpy as jnp
import equinox as eqx

from jax._src.random import KeyArray as PRNGKey
from chex import Array
from typing import List, Tuple, Optional

from jax.nn.initializers import glorot_normal, zeros
from . import nn


class MADE(nn.Dense):

  mask: Array = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               in_dim: int,
               out_dim: int,
               data_dim: int,
               mask_type='hidden',
               act_fn: Optional[nn.ActivationFn] = jax.nn.relu,
               W_init=glorot_normal(),
               b_init=zeros):
    """Create a MADE layer.

    Args:
      key: A JAX PRNGKey used to initialize parameters
      in_dim: The input dimension of the layer.
      out_dim: The output dimension of the layer.
      act_fn: The activation function.
      W_init: The initializer for the weight matrix.
      b_init: The initializer for the bias.
    """
    super().__init__(key, in_dim, out_dim, act_fn=act_fn, W_init=W_init, b_init=b_init)
    assert data_dim > 1, "Must have data dimension > 1."
    assert mask_type in ["input", "output", "hidden"], f"Mask type {mask_type} not 'input', 'output', or 'hidden'."

    max_degree = data_dim - 1
    if mask_type == "hidden":
      in_degrees = (jnp.arange(in_dim) % max_degree) + 1
      out_degrees = (jnp.arange(out_dim) % max_degree) + 1
      self.mask = in_degrees[:, jnp.newaxis] <= out_degrees
    elif mask_type == "input":
      assert in_dim >= data_dim
      assert in_dim % data_dim == 0, "For an input layer in_dim must be a multiple of data_dim."
      in_degrees = jnp.tile(jnp.arange(data_dim)[:, jnp.newaxis], [1, in_dim // data_dim])
      in_degrees = in_degrees.reshape(-1) + 1
      out_degrees = (jnp.arange(out_dim) % max_degree) + 1
      self.mask = in_degrees[:, jnp.newaxis] <= out_degrees
    elif mask_type == "output":
      assert out_dim >= data_dim
      assert out_dim % data_dim == 0, "For output layers, out_dim must be a multiple of data_dim."
      in_degrees = (jnp.arange(in_dim) % max_degree) + 1
      out_degrees = jnp.tile(jnp.arange(data_dim) + 1, [out_dim // data_dim])
      self.mask = in_degrees[:, jnp.newaxis] < out_degrees

  def __call__(self, x: Array) -> Array:
    """Apply the dense layer.

    Args:
      x: the inputs to the layer, must be of shape [..., in_dim].
    Returns:
      output: activation_fn(inputs @ (Mask * W) + b).
    """
    return self.act_fn(jnp.dot(x, self.aff.W * self.mask) + self.aff.b)


class ResMADE(eqx.Module):

  first_layer: MADE
  hidden_layers: List[Tuple[MADE, MADE]]
  last_layer: MADE

  act_fn: nn.ActivationFn = eqx.static_field()
  outputs_per_dim: int = eqx.static_field()
  data_dim: int = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               data_dim: int,
               hidden_dim: int,
               num_res_blocks: int,
               inputs_per_dim: int = 1,
               outputs_per_dim: int = 1,
               act_fn: Optional[nn.ActivationFn] = jax.nn.relu,
               W_init=glorot_normal(),
               b_init=zeros):
    if act_fn is None:
      act_fn = lambda x: x

    self.act_fn = act_fn
    self.outputs_per_dim = outputs_per_dim
    self.data_dim = data_dim
    keys = jax.random.split(key, num=num_res_blocks * 2 + 2)
    self.first_layer = MADE(
            keys[0], inputs_per_dim * data_dim, hidden_dim, data_dim, mask_type='input',
            act_fn=None, W_init=W_init, b_init=b_init)
    hiddens = []
    for i in range(num_res_blocks):
      hiddens.append((
          MADE(keys[i * 2 + 1], hidden_dim, hidden_dim, data_dim, mask_type='hidden',
              act_fn=act_fn, W_init=W_init, b_init=b_init),
          MADE(keys[i * 2 + 2], hidden_dim, hidden_dim, data_dim, mask_type='hidden',
              act_fn=None, W_init=W_init, b_init=b_init)))
    self.hidden_layers = hiddens
    self.last_layer = MADE(keys[-1], hidden_dim, outputs_per_dim * data_dim, data_dim,
            mask_type='output', act_fn=None, W_init=W_init, b_init=b_init)

  def __call__(self, x: Array):
    x = jnp.reshape(x, -1)
    x = self.first_layer(x)
    for layer_a, layer_b in self.hidden_layers:
      # pre-activation resnet
      residual = self.act_fn(x)
      residual = layer_a(residual)
      residual = layer_b(residual)
      x = x + residual
    x = self.act_fn(x)
    x = self.last_layer(x)

    if self.outputs_per_dim > 1:
      out = jnp.reshape(x, [self.outputs_per_dim, self.data_dim]).T
    else:
      out = x
    return out
