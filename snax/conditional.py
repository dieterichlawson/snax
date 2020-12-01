import jax
import jax.numpy as jnp
import jax.tree_util

from typing import NamedTuple, Any

from .base import Module
from .nn import Affine, AffineParams

from jax.nn.initializers import glorot_normal, normal

import tensorflow_probability
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def unflatten_scale(flat_scale, original_dim, min_diag=1e-4):
  out = jnp.zeros([original_dim, original_dim], dtype=flat_scale.dtype)
  out = out.at[jnp.tril_indices(original_dim)].set(flat_scale)
  exp_diag = jnp.exp(jnp.diag(out)) + min_diag
  return out.at[jnp.diag_indices(original_dim)].set(exp_diag)

def vectorize_pytree(*args):
  flat_tree, _ = jax.tree_util.tree_flatten(args)
  flat_vs = [x.flatten() for x in flat_tree]
  return jnp.concatenate(flat_vs, axis=0)


class ConditionalDistributionParams(NamedTuple):

  conditioning_fn_params: Any
  projection_params: AffineParams

def ConditionalDistribution(
        conditioning_fn, 
        dist_fn, 
        event_dim, 
        param_dim,
        W_init=glorot_normal(), 
        b_init=normal()):
  """A module which produces a distribution based on conditioning input.
  
  This module accepts any number of arguments and produces a distribution with 
  parameters that are a function of the input arguments.

  The arguments can be any arity or shape. All arguments are treated
  as a single pytree and flattened into a single vector. That vector is then 
  fed through the conditioning function, and the output is linearly projected
  to the dimension of the parameters of the output distribution.
  """
  
  proj = Affine(param_dim, W_init=W_init, b_init=b_init)

  def init(key, *example_args):
    k1, k2 = jax.random.split(key)
    input = vectorize_pytree(example_args)
    cond_fn_out_dim, cond_fn_params = conditioning_fn.init(k1, input.shape[0])
    _, proj_params = proj.init(k2, cond_fn_out_dim)
    return [event_dim], ConditionalDistributionParams(cond_fn_params, proj_params)

  def apply(params, *args):
    input = vectorize_pytree(args)
    conditioning = conditioning_fn.apply(params.conditioning_fn_params, input)
    raw_params = proj.apply(params.projection_params, conditioning)
    return dist_fn(raw_params)

  return Module(init, apply)

def IsotropicGaussian(
        conditioning_fn, 
        event_dim, 
        min_scale_diag=1e-4,
        W_init=glorot_normal(),
        b_init=normal()):
  """A conditional Gaussian with full covariance matrix.
  
  The distribution mean and covariance are functions of the conditioning set. 
  The covariance matrix is diagonal, and is made positive by exponentiating the
  output of the conditioning function.
  """

  def dist_fn(raw_params):
    mus, raw_log_vars = jnp.split(raw_params, 2)
    vars = jnp.exp(raw_log_vars) + min_scale_diag
    return tfd.MultivariateNormalDiag(loc=mus, scale_diag=jnp.sqrt(vars))
  
  return ConditionalDistribution(conditioning_fn, dist_fn, event_dim, 
                                 2*event_dim, W_init=W_init, b_init=b_init)

def FullCovarianceGaussian(
        conditioning_fn, 
        event_dim, 
        min_scale_diag=1e-4,
        W_init=glorot_normal(),
        b_init=normal()):
  """A conditional Gaussian with full covariance matrix.
  
  The distribution mean and covariance are functions of the conditioning set. 
  The covariance is parameterized as the matrix square of the scale, and the
  scale is parameterized as a lower triangular matrix with positive diagonal
  and unrestricted off-diagonal elements. The diagonal elements are ensured
  to be positive by exponentiating them.
  """
  if covariance is None:
    covariance = jnp.eye(event_dim)

  def dist_fn(raw_params):
    loc = raw_params[:event_dim]
    raw_scale = raw_params[event_dim:]
    scale = unflatten_scale(raw_scale, event_dim, 
                            min_diag=min_scale_diag)
    cov = scale @ scale.T
    return tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
  
  param_dim = event_dim + int((event_dim * (event_dim + 1))/2)
  return ConditionalDistribution(conditioning_fn, dist_fn, event_dim, param_dim, 
                                 W_init=W_init, b_init=b_init)


def FixedCovarianceGaussian(
        conditioning_fn, 
        event_dim,
        covariance=None,
        W_init=glorot_normal(),
        b_init=normal()):
  """A conditional Gaussian with fixed covariance matrix.
  
  The distribution mean is a function of the conditioning set. The covariance 
  matrix is supplied by the user.
  """
  if covariance is None:
    covariance = jnp.eye(event_dim)

  def dist_fn(raw_params):
    return tfd.MultivariateNormalFullCovariance(
        loc=raw_params, covariance_matrix=covariance)
  
  return ConditionalDistribution(conditioning_fn, dist_fn, event_dim, event_dim,
                                 W_init=W_init, b_init=b_init)
