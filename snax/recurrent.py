import jax
import jax.numpy as jnp
import functools

from jax._src.random import KeyArray as PRNGKey
from typing import Tuple, Sequence, Optional, TypeVar, Callable
from chex import Array, ArrayTree
from dataclasses import dataclass

from .base import ParamBase, RecurrentCell, DeepRecurrentModel
from .utils import register_pytree, flip_first_n
from .nn import Linear, Affine, LinearParams, AffineParams

from jax.nn.initializers import glorot_normal, zeros, orthogonal

ActivationFn = Callable[[Array], Array]

@register_pytree
@dataclass
class VanillaRNNState(ParamBase):

  hidden: Array

@register_pytree
@dataclass
class VanillaRNNParams(ParamBase):

  input: LinearParams
  recurrent: AffineParams

def VanillaRNNCell(
        hidden_dim: int,
        activation: ActivationFn = jax.nn.relu,
        input_W_init=glorot_normal(),
        recurrent_W_init=orthogonal(),
        b_init=zeros
        ) -> RecurrentCell[VanillaRNNParams, VanillaRNNState]:
  """Recurrent Neural Network cell."""

  input_layer = Linear(hidden_dim, W_init=input_W_init)
  recurrent_layer = Affine(hidden_dim, W_init=recurrent_W_init, b_init=b_init)

  class VanillaRNNCell:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, VanillaRNNParams]:
      k1, k2 = jax.random.split(key)
      _, input_params = input_layer.init(k1, input_dim)
      _, recurrent_params = recurrent_layer.init(k2, hidden_dim)
      return hidden_dim, VanillaRNNParams(
              input=input_params,
              recurrent=recurrent_params)

    @staticmethod
    def apply(
          params: VanillaRNNParams,
          inputs: Array,
          prev_state: VanillaRNNState) -> Tuple[VanillaRNNState, Array]:
      new_hidden_raw = input_layer.apply(params.input, inputs)
      new_hidden_raw += recurrent_layer.apply(params.recurrent, prev_state.hidden)
      new_hidden = activation(new_hidden_raw)
      new_state = VanillaRNNState(hidden=new_hidden)
      return new_state, new_hidden

    @staticmethod
    def initial_state() -> VanillaRNNState:
      return VanillaRNNState(hidden=jnp.zeros([hidden_dim]))

  return VanillaRNNCell

@register_pytree
@dataclass
class LSTMState(ParamBase):
  hidden: jnp.ndarray
  cell: jnp.ndarray

@register_pytree
@dataclass
class LSTMParams(ParamBase):

  input: LinearParams
  recurrent: AffineParams

def LSTMCell(
        hidden_dim: int,
        activation: ActivationFn = jnp.tanh,
        recurrent_activation: ActivationFn = jax.nn.sigmoid,
        forget_gate_bias_init=1.,
        input_W_init=glorot_normal(),
        recurrent_W_init=orthogonal(),
        b_init=zeros) -> RecurrentCell[LSTMParams, LSTMState]:
  r"""Long short term memory cell, defined by the equations

     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i)
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f)
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o)
     g_t =  \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g)
     c_t = f_t c_{t-1} + i_t g_t
     h_t = o_t \tanh(c_t)
  """

  input_linear = Linear(4 * hidden_dim, W_init=input_W_init)
  recurrent_affine = Affine(4 * hidden_dim, W_init=recurrent_W_init, b_init=b_init)

  class LSTMCell:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, LSTMParams]:
      _, input_params = input_linear.init(key, input_dim)
      _, recurrent_params = recurrent_affine.init(key, hidden_dim)
      return hidden_dim, LSTMParams(input_params, recurrent_params)

    @staticmethod
    def apply(params: LSTMParams,
              inputs: Array,
              prev_state: LSTMState) -> Tuple[LSTMState, Array]:

      input_linear_outs = input_linear.apply(params.input, inputs)
      recurrent_affine_outs = recurrent_affine.apply(params.recurrent, prev_state.hidden)
      raw_outs = input_linear_outs + recurrent_affine_outs
      raw_i, raw_f, raw_o, raw_g = jnp.split(raw_outs, 4)
      raw_f += forget_gate_bias_init

      new_cell = (recurrent_activation(raw_f) * prev_state.cell +
                  recurrent_activation(raw_i) * activation(raw_g))
      new_hidden = recurrent_activation(raw_o) * activation(new_cell)

      new_state = LSTMState(hidden=new_hidden, cell=new_cell)
      return new_state, new_hidden

    @staticmethod
    def initial_state() -> LSTMState:
      return LSTMState(hidden=jnp.zeros([hidden_dim]),
                       cell=jnp.zeros([hidden_dim]))

  return LSTMCell

@register_pytree
@dataclass
class GRUState(ParamBase):

  hidden: jnp.ndarray

@register_pytree
@dataclass
class GRUParams(ParamBase):

  input_to_z: LinearParams
  hidden_to_z: AffineParams
  input_to_r: LinearParams
  hidden_to_r: AffineParams
  input_to_a: LinearParams
  hidden_to_a: AffineParams

def GRUCell(
        hidden_dim: int,
        activation: ActivationFn = jnp.tanh,
        recurrent_activation: ActivationFn = jax.nn.sigmoid,
        input_W_init=glorot_normal(),
        recurrent_W_init=orthogonal(),
        b_init=zeros) -> RecurrentCell[GRUParams, GRUState]:
  r"""Gated recurrent unit cell, defined by the equations

     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z)
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r)
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a)
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
  """
  input_to_z = Linear(hidden_dim, W_init=input_W_init)
  hidden_to_z = Affine(hidden_dim, W_init=recurrent_W_init, b_init=b_init)
  input_to_r = Linear(hidden_dim, W_init=input_W_init)
  hidden_to_r = Affine(hidden_dim, W_init=recurrent_W_init, b_init=b_init)
  input_to_a = Linear(hidden_dim, W_init=input_W_init)
  hidden_to_a = Affine(hidden_dim, W_init=recurrent_W_init, b_init=b_init)

  class GRUCell:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, GRUParams]:
      keys = jax.random.split(key, num=6)
      _, input_to_z_params = input_to_z.init(keys[0], input_dim)
      _, hidden_to_z_params = hidden_to_z.init(keys[1], hidden_dim)
      _, input_to_r_params = input_to_r.init(keys[2], input_dim)
      _, hidden_to_r_params = hidden_to_r.init(keys[3], hidden_dim)
      _, input_to_a_params = input_to_a.init(keys[4], input_dim)
      _, hidden_to_a_params = hidden_to_a.init(keys[5], hidden_dim)
      return hidden_dim, GRUParams(input_to_z_params, hidden_to_z_params,
                                   input_to_r_params, hidden_to_r_params,
                                   input_to_a_params, hidden_to_a_params)

    @staticmethod
    def apply(params: GRUParams,
              inputs: Array,
              prev_state: GRUState) -> Tuple[GRUState, Array]:
      prev_hidden = prev_state.hidden
      z = recurrent_activation(
          input_to_z.apply(params.input_to_z, inputs) +
          hidden_to_z.apply(params.hidden_to_z, prev_hidden))
      r = recurrent_activation(
          input_to_r.apply(params.input_to_r, inputs) +
          hidden_to_r.apply(params.hidden_to_r, prev_hidden))
      a = activation(
          input_to_a.apply(params.input_to_a, inputs) +
          hidden_to_a.apply(params.hidden_to_a, r * prev_hidden))
      new_hidden = (1 - z) * prev_hidden + z * a
      new_state = GRUState(hidden=new_hidden)
      return new_state, new_hidden

    @staticmethod
    def initial_state() -> GRUState:
      return GRUState(hidden=jnp.zeros([hidden_dim]))

  return GRUCell

P = TypeVar('P', bound=ArrayTree)
S = TypeVar('S', bound=ArrayTree)

def RNN(cells: Sequence[RecurrentCell[P, S]]) -> DeepRecurrentModel[P, S]:

  class RNN:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, Sequence[P]]:
      keys = jax.random.split(key, num=len(cells))
      dim = input_dim
      params = []
      for cell, key in zip(cells, keys):
        dim, cell_params = cell.init(key, dim)
        params.append(cell_params)
      return dim, params

    @staticmethod
    def initial_state() -> Sequence[S]:
      return [cell.initial_state() for cell in cells]

    @staticmethod
    def one_step(
        params: Sequence[P],
        inputs: Array,
        prev_state: Sequence[S]) -> Tuple[Sequence[S], Array]:
      new_states = []
      new_out = jnp.array([])
      for cell, cell_prev_state, cell_params in zip(cells, prev_state, params):
        new_state, new_out = cell.apply(cell_params, inputs, cell_prev_state)
        new_states.append(new_state)
        inputs = new_out
      return new_states, new_out

    @staticmethod
    def apply(
        params: Sequence[P],
        inputs: Array,
        length: int,
        initial_state: Optional[Sequence[S]]) -> Tuple[Sequence[S], Array]:

      del length

      def scan_body(
          rnn_state: Sequence[S],
          input: Array) -> Tuple[Sequence[S], Tuple[Sequence[S], Array]]:
        new_state, out = RNN.one_step(params, input, rnn_state)
        return new_state, (new_state, out)

      if initial_state is None:
        initial_state = RNN.initial_state()

      _, (states, outputs) = jax.lax.scan(scan_body, initial_state, inputs)

      return states, outputs

  return RNN

def BidirectionalRNN(
      cells: Sequence[RecurrentCell[P, S]]
    ) -> DeepRecurrentModel[Tuple[P, P], Tuple[S, S]]:

  BiRNNCellState = Tuple[S, S]
  BiRNNCellParams = Tuple[P, P]
  BiRNNState = Sequence[BiRNNCellState]
  BiRNNParams = Sequence[BiRNNCellParams]

  class BidirectionalRNN:

    @staticmethod
    def init(key: PRNGKey, input_dim: int) -> Tuple[int, BiRNNParams]:
      keys = jax.random.split(key, num=len(cells))
      in_dim = input_dim
      params = []
      for cell, key in zip(cells, keys):
        dim, fwd_cell_params = cell.init(key, in_dim)
        _, bwd_cell_params = cell.init(key, in_dim)
        in_dim = dim
        params.append((fwd_cell_params, bwd_cell_params))
        # Bidirectional RNN doubles the 'output' size.
        in_dim = in_dim*2
      return in_dim, params

    @staticmethod
    def initial_state() -> BiRNNState:
      return [(cell.initial_state(), cell.initial_state()) for cell in cells]

    @staticmethod
    def apply_one_layer(
            cell: RecurrentCell,
            params: BiRNNCellParams,
            inputs: Array,
            length: int,
            initial_states: BiRNNCellState) -> Tuple[BiRNNCellState, Array]:

      def scan_fn(
              params: BiRNNCellParams,
              prev_state: BiRNNCellState,
              inputs: Tuple[Array, Array],
              ) -> Tuple[BiRNNCellState, Tuple[BiRNNCellState, Array]]:
        fwd_params, bwd_params = params
        fwd_state, bwd_state = prev_state
        fwd_input, bwd_input = inputs
        new_fwd_state, fwd_out = cell.apply(fwd_params, fwd_input, fwd_state)
        new_bwd_state, bwd_out = cell.apply(bwd_params, bwd_input, bwd_state)
        new_state = (new_fwd_state, new_bwd_state)
        outs = jnp.concatenate([fwd_out, bwd_out], axis=0)
        return new_state, (new_state, outs)

      scan_fn_with_params = functools.partial(scan_fn, params)
      bwd_inputs = flip_first_n(inputs, length)
      _, (states, outs) = jax.lax.scan(scan_fn_with_params, initial_states, (inputs, bwd_inputs))
      return states, outs

    @staticmethod
    def apply(
        params: BiRNNParams,
        inputs: Array,
        length: int,
        initial_state: Optional[BiRNNState]) -> Tuple[BiRNNState, Array]:

      if initial_state is None:
        initial_state = BidirectionalRNN.initial_state()

      states = []
      layer_outs = inputs
      for layer_cell, layer_params, layer_init_states in zip(cells, params, initial_state):
        layer_states, layer_outs = BidirectionalRNN.apply_one_layer(
                layer_cell, layer_params, layer_outs, length, layer_init_states)
        states.append(layer_states)
      return states, layer_outs

  return BidirectionalRNN

def VanillaRNN(
    hidden_dims: Sequence[int],
    activation: ActivationFn = jax.nn.relu,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[VanillaRNNParams, VanillaRNNState]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(VanillaRNNCell(
        hidden_dim,
        activation=activation,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return RNN(cells)

def BidirectionalVanillaRNN(
    hidden_dims: Sequence[int],
    activation: ActivationFn = jax.nn.relu,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[
            Tuple[VanillaRNNParams, VanillaRNNParams],
            Tuple[VanillaRNNState, VanillaRNNState]]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(VanillaRNNCell(
        hidden_dim,
        activation=activation,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return BidirectionalRNN(cells)

def LSTM(
    hidden_dims: Sequence[int],
    activation=jnp.tanh,
    recurrent_activation=jax.nn.sigmoid,
    forget_gate_bias_init=1.,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[LSTMParams, LSTMState]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(LSTMCell(
        hidden_dim,
        activation=activation,
        recurrent_activation=recurrent_activation,
        forget_gate_bias_init=forget_gate_bias_init,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return RNN(cells)

def BidirectionalLSTM(
    hidden_dims: Sequence[int],
    activation=jnp.tanh,
    recurrent_activation=jax.nn.sigmoid,
    forget_gate_bias_init=1.,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[
            Tuple[LSTMParams, LSTMParams], 
            Tuple[LSTMState, LSTMState]]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(LSTMCell(
        hidden_dim,
        activation=activation,
        recurrent_activation=recurrent_activation,
        forget_gate_bias_init=forget_gate_bias_init,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return BidirectionalRNN(cells)

def GRU(
    hidden_dims: Sequence[int],
    activation: ActivationFn = jnp.tanh,
    recurrent_activation: ActivationFn = jax.nn.sigmoid,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[GRUParams, GRUState]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(GRUCell(
        hidden_dim,
        activation=activation,
        recurrent_activation=recurrent_activation,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return RNN(cells)

def BidirectionalGRU(
    hidden_dims: Sequence[int],
    activation: ActivationFn = jnp.tanh,
    recurrent_activation: ActivationFn = jax.nn.sigmoid,
    input_W_init=glorot_normal(),
    recurrent_W_init=orthogonal(),
    b_init=zeros) -> DeepRecurrentModel[Tuple[GRUParams, GRUParams], Tuple[GRUState, GRUState]]:

  cells = []
  for hidden_dim in hidden_dims:
    cells.append(LSTMCell(
        hidden_dim,
        activation=activation,
        recurrent_activation=recurrent_activation,
        input_W_init=input_W_init,
        recurrent_W_init=recurrent_W_init,
        b_init=b_init))

  return BidirectionalRNN(cells)
