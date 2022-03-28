import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass

from jax._src.random import KeyArray as PRNGKey
from typing import TypeVar, Tuple, Callable, Generic, List, Optional
from chex import Array, Scalar

from .utils import flip_first_n, register_dataclass
from .nn import Linear, Affine
from .base import RecurrentCell

from jax.nn.initializers import glorot_normal, zeros, orthogonal

ActivationFn = Callable[[Array], Array]

@register_dataclass
@dataclass
class VanillaRNNState:
  hidden: Array


class VanillaRNNCell(eqx.Module):

  input_layer: Linear
  recurrent_layer: Affine
  act_fn: ActivationFn = eqx.static_field()
  hidden_dim: int = eqx.static_field()
  out_dim: int = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dim: int,
               act_fn: ActivationFn = jax.nn.relu,
               input_W_init=glorot_normal(),
               recurrent_W_init=orthogonal(),
               b_init=zeros):
    k1, k2 = jax.random.split(key)
    self.input_layer = Linear(k1,
                              input_dim,
                              hidden_dim,
                              W_init=input_W_init)
    self.recurrent_layer = Affine(k2,
                                  hidden_dim,
                                  hidden_dim,
                                  W_init=recurrent_W_init,
                                  b_init=b_init)
    self.act_fn = act_fn
    self.hidden_dim = hidden_dim
    self.out_dim = hidden_dim

  def __call__(self,
               prev_state: VanillaRNNState,
               inputs: Array
               ) -> Tuple[VanillaRNNState, Array]:
    new_hidden_raw = self.input_layer(inputs)
    new_hidden_raw += self.recurrent_layer(prev_state.hidden)
    new_hidden = self.act_fn(new_hidden_raw)
    new_state = VanillaRNNState(hidden=new_hidden)
    return new_state, new_hidden

  def initial_state(self) -> VanillaRNNState:
    return VanillaRNNState(hidden=jnp.zeros(self.hidden_dim))


@register_dataclass
@dataclass
class LSTMState:
  hidden: Array
  cell: Array


class LSTMCell(eqx.Module):

  input_linear: Linear
  recurrent_affine: Affine

  hidden_dim: int = eqx.static_field()
  out_dim: int = eqx.static_field()
  act_fn: Callable = eqx.static_field()
  recurrent_act_fn: Callable = eqx.static_field()
  forget_gate_bias_init: int = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dim: int,
               act_fn: Callable = jnp.tanh,
               recurrent_act_fn: Callable = jax.nn.sigmoid,
               forget_gate_bias_init=1.,
               input_W_init=glorot_normal(),
               recurrent_W_init=orthogonal(),
               b_init=zeros):
    k1, k2 = jax.random.split(key)
    self.input_linear = Linear(k1,
                               input_dim,
                               4 * hidden_dim,
                               W_init=input_W_init)
    self.recurrent_affine = Affine(k2,
                                   hidden_dim,
                                   4 * hidden_dim,
                                   W_init=recurrent_W_init,
                                   b_init=b_init)
    self.hidden_dim = hidden_dim
    self.out_dim = hidden_dim
    self.act_fn = act_fn
    self.recurrent_act_fn = recurrent_act_fn
    self.forget_gate_bias_init = forget_gate_bias_init

  def __call__(self,
               prev_state: LSTMState,
               inputs: Array
               ) -> Tuple[LSTMState, Array]:
    input_linear_outs = self.input_linear(inputs)
    recurrent_affine_outs = self.recurrent_affine(prev_state.hidden)
    raw_outs = input_linear_outs + recurrent_affine_outs
    raw_i, raw_f, raw_o, raw_g = jnp.split(raw_outs, 4)
    raw_f += self.forget_gate_bias_init

    new_cell = (self.recurrent_act_fn(raw_f) * prev_state.cell +
                self.recurrent_act_fn(raw_i) * self.act_fn(raw_g))
    new_hidden = self.recurrent_act_fn(raw_o) * self.act_fn(new_cell)

    new_state = LSTMState(hidden=new_hidden, cell=new_cell)
    return new_state, new_hidden

  def initial_state(self) -> LSTMState:
    return LSTMState(hidden=jnp.zeros([self.hidden_dim]),
                     cell=jnp.zeros([self.hidden_dim]))


@register_dataclass
@dataclass
class GRUState:
  hidden: Array


class GRUCell(eqx.Module):

  input_to_z: Linear
  hidden_to_z: Affine
  input_to_r: Linear
  hidden_to_r: Affine
  input_to_a: Linear
  hidden_to_a: Affine

  hidden_dim: int = eqx.static_field()
  out_dim: int = eqx.static_field()
  act_fn: Callable = eqx.static_field()
  recurrent_act_fn: Callable = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dim: int,
               act_fn: Callable = jnp.tanh,
               recurrent_act_fn: Callable = jax.nn.sigmoid,
               input_W_init=glorot_normal(),
               recurrent_W_init=orthogonal(),
               b_init=zeros):
    keys = jax.random.split(key, 6)
    self.input_to_z = Linear(
        keys[0], input_dim, hidden_dim, W_init=input_W_init)
    self.input_to_r = Linear(
        keys[1], input_dim, hidden_dim, W_init=input_W_init)
    self.input_to_a = Linear(
        keys[2], input_dim, hidden_dim, W_init=input_W_init)
    self.hidden_to_z = Affine(
        keys[3], hidden_dim, hidden_dim, W_init=recurrent_W_init, b_init=b_init)
    self.hidden_to_r = Affine(
        keys[4], hidden_dim, hidden_dim, W_init=recurrent_W_init, b_init=b_init)
    self.hidden_to_a = Affine(
        keys[5], hidden_dim, hidden_dim, W_init=recurrent_W_init, b_init=b_init)

    self.hidden_dim = hidden_dim
    self.out_dim = hidden_dim
    self.act_fn = act_fn
    self.recurrent_act_fn = recurrent_act_fn

  def __call__(self,
               prev_state: GRUState,
               inputs: Array
               ) -> Tuple[GRUState, Array]:
    prev_hidden = prev_state.hidden
    z = self.recurrent_act_fn(
        self.input_to_z(inputs) + self.hidden_to_z(prev_hidden))
    r = self.recurrent_act_fn(
        self.input_to_r(inputs) + self.hidden_to_r(prev_hidden))
    a = self.act_fn(
        self.input_to_a(inputs) + self.hidden_to_a(r * prev_hidden))
    new_hidden = (1 - z) * prev_hidden + z * a
    new_state = GRUState(hidden=new_hidden)
    return new_state, new_hidden

  def initial_state(self) -> GRUState:
    return GRUState(hidden=jnp.zeros([self.hidden_dim]))


StateType = TypeVar("StateType")
RNNCellConstructor = Callable[[PRNGKey, int, int], RecurrentCell[StateType]]


class RNN(eqx.Module, Generic[StateType]):

  cells: List[RecurrentCell[StateType]]

  in_dim: int = eqx.static_field()
  out_dim: int = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dims: List[int],
               cell_constructor: RNNCellConstructor[StateType]):
    dims = zip([input_dim] + list(hidden_dims), hidden_dims)
    cells = []
    self.in_dim = input_dim
    for in_dim, out_dim in dims:
      key, subkey = jax.random.split(key)
      cells.append(cell_constructor(subkey, in_dim, out_dim))
    self.out_dim = out_dim
    self.cells = cells

  def initial_state(self) -> List[StateType]:
    return [cell.initial_state() for cell in self.cells]

  def one_step(
          self,
          prev_state: List[StateType],
          inputs: Array) -> Tuple[List[StateType], Array]:
    new_states = []
    new_out = jnp.array([])
    for cell, cell_prev_state in zip(self.cells, prev_state):
      new_state, new_out = cell(cell_prev_state, inputs)
      new_states.append(new_state)
      inputs = new_out
    return new_states, new_out

  def __call__(
          self,
          inputs: Array,
          initial_state : Optional[List[StateType]] = None,
          initial_state_t: Optional[Scalar] = None) -> Tuple[List[StateType], Array]:

    if initial_state is None:
      initial_state = self.initial_state()

    def scan_body(
            carry: Tuple[List[StateType], int],
            input: Array
            ) -> Tuple[Tuple[List[StateType], int], Tuple[List[StateType], Array]]:
      prev_state, t = carry

      if initial_state_t is not None:
        prev_state = jax.lax.cond(
                jnp.equal(initial_state_t, t),
                lambda _: initial_state,
                lambda _: prev_state,
                None)

      new_state, out = self.one_step(prev_state, input)
      return (new_state, t+1), (new_state, out)


    _, (states, outputs) = jax.lax.scan(scan_body, (initial_state, 0), inputs)

    return states, outputs


BiRNNCellState = Tuple[StateType, StateType]


class BiRNNCell(eqx.Module, Generic[StateType]):

  fwd_cell: RecurrentCell[StateType]
  bwd_cell: RecurrentCell[StateType]

  out_dim: int = eqx.static_field()

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dim: int,
               cell_constructor: RNNCellConstructor[StateType]):
    k1, k2 = jax.random.split(key)
    self.fwd_cell = cell_constructor(k1, input_dim, hidden_dim)
    self.bwd_cell = cell_constructor(k2, input_dim, hidden_dim)
    self.out_dim = self.fwd_cell.out_dim + self.bwd_cell.out_dim

  def initial_state(self) -> BiRNNCellState[StateType]:
    return (self.fwd_cell.initial_state(), self.bwd_cell.initial_state())

  def __call__(self,
               prev_state: BiRNNCellState[StateType],
               inputs: Tuple[Array, Array]
               ) -> Tuple[BiRNNCellState[StateType], Array]:
    fwd_state, bwd_state = prev_state
    fwd_input, bwd_input = inputs
    new_fwd_state, fwd_out = self.fwd_cell(fwd_state, fwd_input)
    new_bwd_state, bwd_out = self.bwd_cell(bwd_state, bwd_input)
    new_state = (new_fwd_state, new_bwd_state)
    outs = jnp.concatenate([fwd_out, bwd_out], axis=0)
    return new_state, outs


BiRNNState = List[BiRNNCellState[StateType]]


class BiRNN(eqx.Module, Generic[StateType]):

  cells: List[BiRNNCell[StateType]]

  def __init__(self,
               key: PRNGKey,
               input_dim: int,
               hidden_dims: List[int],
               cell_constructor: RNNCellConstructor[StateType]):
    in_dims = [input_dim] + [x*2 for x in hidden_dims]
    dims = zip(in_dims, hidden_dims)
    self.cells = []
    for in_dim, hid_dim in dims:
      key, sk = jax.random.split(key)
      self.cells.append(BiRNNCell(sk, in_dim, hid_dim, cell_constructor))

  def initial_state(self) -> BiRNNState[StateType]:
    return [cell.initial_state() for cell in self.cells]

  def apply_one_layer(
      self,
      l: int,
      inputs: Array,
      length: int,
      initial_state: BiRNNCellState[StateType]
      ) -> Tuple[BiRNNCellState[StateType], Array]:
    assert l >= 0 and l < len(self.cells), "Tried to apply non-existent layer"

    def scan_fn(
        prev_state: BiRNNCellState[StateType],
        inputs: Tuple[Array, Array]
        ) -> Tuple[BiRNNCellState[StateType], Tuple[BiRNNCellState[StateType], Array]]:
      new_state, outs = self.cells[l](prev_state, inputs)
      return new_state, (new_state, outs)

    bwd_inputs = flip_first_n(inputs, length)
    _, (states, outs) = jax.lax.scan(
        scan_fn, initial_state, (inputs, bwd_inputs))
    return states, outs

  def __call__(
      self,
      inputs: Array,
      length: int,
      initial_state: Optional[BiRNNState[StateType]] = None,
      ) -> Tuple[BiRNNState[StateType], Array]:

    if initial_state is None:
      initial_state = self.initial_state()

    states = []
    layer_outs = inputs
    for i in range(len(self.cells)):
      layer_states, layer_outs = self.apply_one_layer(
          i, layer_outs, length, initial_state[i]
      )
      states.append(layer_states)
    return states, layer_outs


class VanillaRNN(RNN[VanillaRNNState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jax.nn.relu,
          input_W_init=glorot_normal(),
          recurrent_W_init=glorot_normal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> VanillaRNNCell:
      return VanillaRNNCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)


class LSTM(RNN[LSTMState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jnp.tanh,
          recurrent_act_fn: ActivationFn = jax.nn.sigmoid,
          forget_gate_bias_init=1.,
          input_W_init=glorot_normal(),
          recurrent_W_init=orthogonal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> LSTMCell:
      return LSTMCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              recurrent_act_fn=recurrent_act_fn,
              forget_gate_bias_init=forget_gate_bias_init,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)


class GRU(RNN[GRUState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jnp.tanh,
          recurrent_act_fn: ActivationFn = jax.nn.sigmoid,
          input_W_init=glorot_normal(),
          recurrent_W_init=orthogonal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> GRUCell:
      return GRUCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              recurrent_act_fn=recurrent_act_fn,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)


class BiVanillaRNN(BiRNN[VanillaRNNState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jax.nn.relu,
          input_W_init=glorot_normal(),
          recurrent_W_init=glorot_normal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> VanillaRNNCell:
      return VanillaRNNCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)


class BiLSTM(BiRNN[LSTMState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jnp.tanh,
          recurrent_act_fn: ActivationFn = jax.nn.sigmoid,
          forget_gate_bias_init=1.,
          input_W_init=glorot_normal(),
          recurrent_W_init=orthogonal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> LSTMCell:
      return LSTMCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              recurrent_act_fn=recurrent_act_fn,
              forget_gate_bias_init=forget_gate_bias_init,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)


class BiGRU(BiRNN[GRUState]):

  def __init__(
          self,
          key: PRNGKey,
          input_dim: int,
          hidden_dims: List[int],
          act_fn: ActivationFn = jnp.tanh,
          recurrent_act_fn: ActivationFn = jax.nn.sigmoid,
          input_W_init=glorot_normal(),
          recurrent_W_init=orthogonal(),
          b_init=zeros):

    def cell_constructor(key: PRNGKey, input_dim: int, hidden_dim: int) -> GRUCell:
      return GRUCell(
              key,
              input_dim,
              hidden_dim,
              act_fn=act_fn,
              recurrent_act_fn=recurrent_act_fn,
              input_W_init=input_W_init,
              recurrent_W_init=recurrent_W_init,
              b_init=b_init)
    super().__init__(key, input_dim, hidden_dims, cell_constructor)
