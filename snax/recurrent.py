import jax
import jax.numpy as jnp

import typing
from typing import NamedTuple, List

from .base import Module
from .nn import Affine, Linear, AffineParams, LinearParams

from jax.nn.initializers import glorot_normal, normal, ones, zeros

class RNNState(NamedTuple):

  hidden: jnp.ndarray

class RNNParams(NamedTuple):

  input_to_hidden: LinearParams
  hidden_to_hidden: AffineParams

def RNN(hidden_dim, W_init=glorot_normal(), b_init=normal(), activation=jax.nn.relu):
  """Recurrent Neural Network cell."""

  input_to_hidden = Linear(hidden_dim, W_init=W_init)
  hidden_to_hidden = Affine(hidden_dim, W_init=W_init, b_init=b_init)

  def init(key, input_dim):
    output_shape = hidden_dim
    k1, k2 = jax.random.split(key)
    _, input_to_hidden_params = input_to_hidden.init(k1, input_dim)
    _, hidden_to_hidden_params = hidden_to_hidden.init(k2, hidden_dim)
    return [hidden_dim], RNNParams(input_to_hidden_params, hidden_to_hidden_params)

  def apply(params, inputs, prev_state, **kwargs):
    new_hidden_raw = (input_to_hidden.apply(params.input_to_hidden, inputs) +
                      hidden_to_hidden.apply(
                          params.hidden_to_hidden, prev_state.hidden))
    new_hidden = activation(new_hidden_raw)
    new_state = RNNState(hidden=new_hidden)
    return new_state, new_hidden

  def initial_state():
    return RNNState(hidden=jnp.zeros([hidden_dim]))

  return Module(init, apply, initial_state)

class LSTMState(NamedTuple):

  hidden: jnp.ndarray
  cell: jnp.ndarray

class LSTMParams(NamedTuple):

  affine: AffineParams

def LSTM(hidden_dim, W_init=glorot_normal(), b_init=normal()):
  """Long short term memory cell, defined by the equations

     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i)
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f)
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o)
     g_t =  \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g)
     c_t = f_t c_{t-1} + i_t g_t
     h_t = o_t \tanh(c_t)
  """

  affine_fn = Affine(4 * hidden_dim, W_init=W_init, b_init=b_init)

  def init(key, input_dim):
    keys = jax.random.split(key, num=8)
    _, params = affine_fn.init(key, input_dim + hidden_dim)
    return [hidden_dim], LSTMParams(params)

  def apply(params, inputs, prev_state, **kwargs):
    prev_hidden = prev_state.hidden
    prev_cell = prev_state.cell

    affine_outs = affine_fn.apply(
        params.affine, jnp.concatenate([prev_hidden, inputs], axis=0))

    raw_i, raw_f, raw_o, raw_g = jnp.split(affine_outs, 4)

    new_cell = (jax.nn.sigmoid(raw_f) * prev_cell +
                jax.nn.sigmoid(raw_i) * jnp.tanh(raw_g))
    new_hidden = jax.nn.sigmoid(raw_o) * jnp.tanh(new_cell)

    new_state = LSTMState(hidden=new_hidden, cell=new_cell)
    return new_state, new_hidden

  def initial_state():
    return LSTMState(hidden=jnp.zeros([hidden_dim]),
                     cell=jnp.zeros([hidden_dim]))

  return Module(init, apply, initial_state)

class GRUState(NamedTuple):

  hidden: jnp.ndarray

class GRUParams(NamedTuple):

  input_to_z: LinearParams
  hidden_to_z: AffineParams
  input_to_r: LinearParams
  hidden_to_r: AffineParams
  input_to_a: LinearParams
  hidden_to_a: AffineParams

def GRU(hidden_dim, W_init=glorot_normal(), b_init=normal()):
  """Gated recurrent unit cell, defined by the equations

     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z)
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r)
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a)
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
  """
  input_to_z = Linear(hidden_dim, W_init=W_init)
  hidden_to_z = Affine(hidden_dim, W_init=W_init, b_init=b_init)
  input_to_r = Linear(hidden_dim, W_init=W_init)
  hidden_to_r = Affine(hidden_dim, W_init=W_init, b_init=b_init)
  input_to_a = Linear(hidden_dim, W_init=W_init)
  hidden_to_a = Affine(hidden_dim, W_init=W_init, b_init=b_init)

  def init(key, input_dim):
    keys = jax.random.split(key, num=6)
    _, input_to_z_params = input_to_z.init(keys[0], input_dim)
    _, hidden_to_z_params = hidden_to_z.init(keys[1], hidden_dim)
    _, input_to_r_params = input_to_r.init(keys[2], input_dim)
    _, hidden_to_r_params = hidden_to_r.init(keys[3], hidden_dim)
    _, input_to_a_params = input_to_a.init(keys[4], input_dim)
    _, hidden_to_a_params = hidden_to_a.init(keys[5], hidden_dim)
    return [hidden_dim], GRUParams(input_to_z_params, hidden_to_z_params,
                                   input_to_r_params, hidden_to_r_params,
                                   input_to_a_params, hidden_to_a_params)

  def apply(params, inputs, prev_state, **kwargs):
    prev_hidden = prev_state.hidden

    z = jax.nn.sigmoid(
        input_to_z.apply(params.input_to_z, inputs) +
        hidden_to_z.apply(params.hidden_to_z, prev_hidden))
    r = jax.nn.sigmoid(
        input_to_r.apply(params.input_to_r, inputs) +
        hidden_to_r.apply(params.hidden_to_r, prev_hidden))
    a = jnp.tanh(
        input_to_a.apply(params.input_to_a, inputs) +
        hidden_to_a.apply(params.hidden_to_a, r * prev_hidden))
    new_hidden = (1 - z) * prev_hidden + z * a
    new_state = GRUState(hidden=new_hidden)
    return new_state, new_hidden

  def initial_state():
    return GRUState(hidden=jnp.zeros([hidden_dim]))

  return Module(init, apply, initial_state)

def DeepRNN(cell_type, hidden_dims, W_init=glorot_normal(), b_init=normal()):
  """Deep RNN cell, a wrapper for a stack of RNNs."""

  cells = [cell_type(h, W_init=W_init, b_init=b_init) for h in hidden_dims]

  def init(key, input_dim):
    keys = jax.random.split(key, num=len(cells))
    in_dims = [input_dim] + hidden_dims[:-1]
    params = []
    for cell, key, dim in zip(cells, keys, in_dims):
      params.append(cell.init(key, dim)[1])
    return [hidden_dims[-1]], params

  def apply(cells_params, inputs, prev_states, **kwargs):
    new_states = []
    for cell, prev_state, params in zip(cells, prev_states, cells_params):
      new_state, new_out = cell.apply(params, inputs, prev_state)
      new_states.append(new_state)
      inputs = new_out
    return new_states, new_out

  def initial_state():
    return [cell.initial_state() for cell in cells]

  return Module(init, apply, initial_state)

def dynamic_unroll(params, rnn_apply, inputs, initial_state):

  def scan_body(rnn_state, input):
    new_state, out = rnn_apply(params, input, rnn_state)
    return new_state, (new_state, out)

  _, outs = jax.lax.scan(scan_body, initial_state, inputs)
  states, outputs = outs
  return states, outputs
