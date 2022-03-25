import jax.numpy as jnp
import equinox as eqx
from typing import Callable
import tensorflow_probability.substrates.jax as tfp

tf = tfp.tf2jax
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions

def L_from_raw(raw_L, diag_min=1e-6):
  """
  Convert an unconstrained matrix (log-terms on the diagonal + lower triangular)
  into a lower triangular matrix, for which L'L is the full precision matrix.
  :param raw_L:
  :param diag_min:
  :return:
  """
  scale_diag = jnp.maximum(jnp.exp(jnp.diag(raw_L)), diag_min)
  L = jnp.diag(scale_diag) + jnp.tril(raw_L, -1)
  return L


class ConditionalGaussianFullRankPrecision(eqx.Module):

  conditioner: eqx.Module

  min_scale_diag: float = eqx.static_field()
  data_dim: int = eqx.static_field()

  def __init__(
          self,
          key,
          in_dim: int,
          data_dim: int,
          conditioner_constructor: Callable,
          min_scale_diag: float = 1e-4):
    out_dim = int(data_dim ** 2 + data_dim)
    self.conditioner = conditioner_constructor(key, in_dim, out_dim)
    self.min_scale_diag = min_scale_diag
    self.data_dim = data_dim

  def __call__(self, inputs):
    raw_out = self.conditioner(inputs)
    loc = raw_out[:self.data_dim]
    raw_prec_L = jnp.reshape(raw_out[self.data_dim:], [self.data_dim, self.data_dim])
    prec_L = L_from_raw(raw_prec_L, diag_min=jnp.sqrt(self.min_scale_diag))
    prec_factor = tf.linalg.LinearOperatorLowerTriangular(prec_L)
    dist = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
            loc=loc, precision_factor=prec_factor)
    return dist, (loc, prec_L)

class ConditionalGaussianDiagPrecision(eqx.Module):

  conditioner = eqx.Module

  min_scale_diag: float = eqx.static_field()
  data_dim: int = eqx.static_field()

  def __init__(
          self,
          key,
          in_dim: int,
          data_dim: int,
          conditioner_constructor: Callable,
          min_scale_diag: float = 1e-4):
    self.conditioner = conditioner_constructor(key, in_dim, 2 * data_dim)
    self.min_scale_diag = min_scale_diag
    self.data_dim = data_dim

  def __call__(self, inputs):
    raw_out = self.conditioner(inputs)
    loc = raw_out[:self.data_dim]
    raw_prec_diag = raw_out[self.data_dim:]
    prec_diag = jnp.maximum(jnp.exp(raw_prec_diag), self.min_scale_diag)
    prec_factor = tf.linalg.LinearOperatorDiag(prec_diag)
    dist = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
            loc=loc, precision_factor=prec_factor)
    return dist, (loc, prec_diag)
