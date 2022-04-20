import jax
import jax.numpy as jnp
import equinox as eqx

from jax._src.random import KeyArray as PRNGKey
from chex import Array
from typing import List, Callable, Optional

from jax.nn.initializers import glorot_normal, zeros

ActivationFn = Callable[[Array], Array]

class Linear(eqx.Module):
  """A linear layer parameterized by a single weight matrix, W."""

  W: Array

  def __init__(self,
               key: PRNGKey,
               in_dim: int,
               out_dim: int,
               W_init=glorot_normal()):
    """Create a linear layer.

    Args:
      key: A JAX PRNGKey used to initialize the weight matrix.
      in_dim: The input dimension of the layer.
      out_dim: The output dimension of the layer.
      W_init: The initializer for the weight matrix.
    """
    self.W = W_init(key, (in_dim, out_dim))

  def __call__(self, inputs: Array) -> Array:
    """Apply the linear layer.

    Args:
      inputs: the inputs to the layer, must be of shape [..., in_dim].
    Returns:
      output: inputs @ W.
    """
    return jnp.dot(inputs, self.W)

class Affine(eqx.Module):
  """An affine layer parameterized by a weight W and bias b."""

  W: Array
  b: Array

  def __init__(self,
               key: PRNGKey,
               in_dim: int,
               out_dim: int,
               W_init=glorot_normal(),
               b_init=zeros):
    """Create an affine layer.

    Args:
      key: A JAX PRNGKey used to initialize parameters
      in_dim: The input dimension of the layer.
      out_dim: The output dimension of the layer.
      W_init: The initializer for the weight matrix.
      b_init: The initializer for the bias.
    """
    self.W = W_init(key, (in_dim, out_dim))
    self.b = b_init(key, (out_dim,))

  def __call__(self, inputs: Array) -> Array:
    """Apply the affine layer.

    Args:
      inputs: the inputs to the layer, must be of shape [..., in_dim].
    Returns:
      output: inputs @ W + b.
    """

    return jnp.dot(inputs, self.W) + self.b

class Dense(eqx.Module):
  """A dense layer, with a weight W, bias b, and activation function."""

  aff: Affine
  act_fn: ActivationFn = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               in_dim: int,
               out_dim: int,
               act_fn: Optional[ActivationFn] = jax.nn.relu,
               W_init=glorot_normal(),
               b_init=zeros):
    """Create a dense layer.

    Args:
      key: A JAX PRNGKey used to initialize parameters
      in_dim: The input dimension of the layer.
      out_dim: The output dimension of the layer.
      act_fn: The activation function.
      W_init: The initializer for the weight matrix.
      b_init: The initializer for the bias.
    """

    self.aff = Affine(key, in_dim, out_dim, W_init=W_init, b_init=b_init)
    if act_fn is None:
      act_fn = lambda x: x
    self.act_fn = act_fn

  def __call__(self, inputs: Array) -> Array:
    """Apply the dense layer.

    Args:
      inputs: the inputs to the layer, must be of shape [..., in_dim].
    Returns:
      output: activation_fn(inputs @ W + b).
    """

    return self.act_fn(self.aff(inputs))

class MLP(eqx.Module):
  """A multi-layer perceptron made of a stack of dense layers."""

  layers: List[Dense]

  def __init__(self,
               key: PRNGKey,
               in_dim: int,
               layer_dims: List[int],
               act_fn: ActivationFn,
               final_act_fn: ActivationFn= lambda x: x,
               W_init=glorot_normal(),
               b_init=zeros):
    """Create an MLP.

    Args:
      key: A JAX PRNGKey used to initialize parameters
      in_dim: The input dimension of the layer.
      layer_dims: The output dimensions of each layer, a list of ints.
      act_fn: The activation function for the hidden layers.
      final_act_fn: The activation function for the final layer.
      W_init: The initializer for the weight matrix.
      b_init: The initializer for the bias.
    """

    self.layers = []
    dims = [in_dim] + layer_dims
    dim_pairs = list(zip(dims, dims[1:]))
    for in_d, out_d in dim_pairs[:-1]:
      key, subkey = jax.random.split(key)
      self.layers.append(
          Dense(subkey,
                in_d,
                out_d,
                act_fn=act_fn,
                W_init=W_init,
                b_init=b_init))
    final_in_d, final_out_d = dim_pairs[-1]
    self.layers.append(
        Dense(key,
              final_in_d,
              final_out_d,
              act_fn=final_act_fn,
              W_init=W_init,
              b_init=b_init))

  def __call__(self, inputs: Array) -> Array:
    """Apply the MLP.

    Args:
      inputs: The inputs to the layer, must be of shape [..., in_dim].
    Returns:
      output: The MLP applied to the inputs.
    """
    for l in self.layers:
      inputs = l(inputs)
    return inputs
