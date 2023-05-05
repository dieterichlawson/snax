import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from chex import Array

from . import recurrent
from . import utils

def test_basic_vanilla_rnn(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.VanillaRNN(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs)

  states, out = jax.jit(call)(rnn, inputs)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)

def test_basic_vanilla_birnn(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiVanillaRNN(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs, seq_len-1, None)

  states, out = jax.jit(call)(rnn, inputs)
  assert out.shape == (seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (seq_len, size)
    assert bwd_state.hidden.shape == (seq_len, size)

def test_basic_lstm(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.LSTM(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs)

  states, out = jax.jit(call)(rnn, inputs)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)
    assert state.cell.shape == (seq_len, size)

def test_basic_bilstm(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiLSTM(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs, seq_len-1, None)

  states, out = jax.jit(call)(rnn, inputs)

  assert out.shape == (seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (seq_len, size)
    assert fwd_state.cell.shape == (seq_len, size)
    assert bwd_state.hidden.shape == (seq_len, size)
    assert bwd_state.cell.shape == (seq_len, size)

def test_basic_gru(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.GRU(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs)

  states, out = jax.jit(call)(rnn, inputs)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)

def test_basic_bigru(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiGRU(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs, seq_len-1, None)

  states, out = jax.jit(call)(rnn, inputs)
  assert out.shape == (seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (seq_len, size)
    assert bwd_state.hidden.shape == (seq_len, size)

def test_batch_gru(
        batch_size=4,
        hidden_sizes=[3,4,5],
        input_size=2,
        seq_len=11,
        lengths=[1,11,0,9]):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.GRU(key, input_size, hidden_sizes)
  inputs = jnp.ones([batch_size, seq_len, input_size])
  lengths =  jnp.array(lengths)

  def call(model, inputs):
    return model(inputs)

  states, out = jax.vmap(call, in_axes=(None, 0))(rnn, inputs)
  assert out.shape == (batch_size, seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (batch_size, seq_len, size)

def test_batch_bigru(
        batch_size=4,
        hidden_sizes=[3,4,5],
        input_size=2,
        seq_len=11,
        lengths=[1,11,0,9]):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiGRU(key, input_size, hidden_sizes)
  inputs = jnp.ones([batch_size, seq_len, input_size])
  lengths =  jnp.array(lengths)

  def call(model, inputs, lengths):
    return model(inputs, lengths, None)

  states, out = jax.vmap(call, in_axes=(None, 0, 0))(rnn, inputs, lengths)

  assert out.shape == (batch_size, seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (batch_size, seq_len, size)
    assert bwd_state.hidden.shape == (batch_size, seq_len, size)

def test_identity_vanilla_rnn(num_layers=3, input_size=2, seq_len=5):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.VanillaRNN(
          key,
          input_size,
          [input_size] * (num_layers-1),
          act_fn=jax.nn.relu,
          input_W_init=utils.identity_init,
          recurrent_W_init=utils.identity_init,
          b_init=jax.nn.initializers.zeros)

  inputs = jnp.arange(seq_len * input_size).reshape([seq_len, input_size])
  states, out = rnn(inputs)
  state_check = inputs
  for state in states:
    state_check = jnp.cumsum(state_check, axis=0)
    assert jnp.allclose(state.hidden, state_check)
  assert jnp.allclose(out, state_check)


def test_identity_reverse_vanilla_rnn(num_layers=3, input_size=2, seq_len=5):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.VanillaRNN(
          key,
          input_size,
          [input_size] * (num_layers-1),
          act_fn=jax.nn.relu,
          input_W_init=utils.identity_init,
          recurrent_W_init=utils.identity_init,
          b_init=jax.nn.initializers.zeros)

  inputs = jnp.arange(seq_len * input_size).reshape([seq_len, input_size])
  states, out = rnn(inputs, reverse=True)
  state_check = jnp.flip(inputs, axis=0)
  for state in states:
    state_check = jnp.cumsum(state_check, axis=0)
    assert jnp.allclose(state.hidden, jnp.flip(state_check, axis=0)), f"{state_check}, {state.hidden}"
  assert jnp.allclose(out, jnp.flip(state_check, axis=0))


def test_identity_lstmcell():
  input_size=1
  key = jax.random.PRNGKey(0)
  cell = recurrent.LSTMCell(
          key,
          input_size,
          input_size,
          act_fn=jax.nn.relu,
          recurrent_act_fn=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  inputs = jnp.array([2.])
  state, _ = cell(cell.initial_state(), inputs)
  assert jnp.allclose(state.hidden, jnp.array([2**3]))
  assert jnp.allclose(state.cell, jnp.array([2**2]))

def test_identity_lstm():
  input_size=1
  seq_len=4
  key = jax.random.PRNGKey(0)
  rnn = recurrent.LSTM(
          key,
          input_size,
          [input_size],
          act_fn=jax.nn.relu,
          recurrent_act_fn=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  inputs = jnp.arange(seq_len * input_size).reshape([seq_len, input_size])
  states, _ = rnn(inputs)
  states = states[0]
  true_hiddens = jnp.array([0, 1, 36, 39*(39*12 + 39**2)], dtype=jnp.float32)
  true_cells = jnp.array([0, 1, 12, 39*12 + 39**2], dtype=jnp.float32)
  assert jnp.allclose(states.hidden, true_hiddens.reshape(states.hidden.shape))
  assert jnp.allclose(states.cell, true_cells.reshape(states.cell.shape))

def test_identity_bilstm():
  input_size=1
  seq_len=4
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiLSTM(
          key,
          input_size,
          [input_size],
          act_fn=jax.nn.relu,
          recurrent_act_fn=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  inputs = jnp.array([1., 1., 1., 0.]).reshape([seq_len, input_size])
  states, outs = rnn(inputs, seq_len, None)
  fwd_state, bwd_state = states[0]
  true_fwd_cells = jnp.array([1, 6, 247, 11103638], dtype=jnp.float32)
  true_fwd_hiddens = jnp.array([1, 12, 3211, 35653781618], dtype=jnp.float32)
  true_bwd_cells = jnp.array([247, 6, 1, 0], dtype=jnp.float32)
  true_bwd_hiddens = jnp.array([3211, 12, 1, 0], dtype=jnp.float32)
  assert jnp.allclose(fwd_state.hidden, true_fwd_hiddens.reshape(fwd_state.hidden.shape))
  assert jnp.allclose(fwd_state.cell, true_fwd_cells.reshape(fwd_state.hidden.shape))
  assert jnp.allclose(bwd_state.hidden, true_bwd_hiddens.reshape(bwd_state.cell.shape))
  assert jnp.allclose(bwd_state.cell, true_bwd_cells.reshape(bwd_state.cell.shape))
  true_out = jnp.stack([true_fwd_hiddens, true_bwd_hiddens], axis=1)
  assert jnp.allclose(outs, true_out)



def test_deep_identity_bilstm():
  input_size=1
  seq_len=4
  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiLSTM(
          key,
          input_size,
          [input_size, input_size],
          act_fn=jax.nn.relu,
          recurrent_act_fn=jax.nn.relu,
          input_W_init=jax.nn.initializers.ones,
          recurrent_W_init=jax.nn.initializers.ones,
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  inputs = jnp.array([-1., 0., .5, 0.]).reshape([seq_len, input_size])
  states, outs = rnn(inputs, seq_len, None)
  fwd_state_1, bwd_state_1 = states[0]
  true_fwd_cells_1 = jnp.array([0., 0., 0.25, 0.046875], dtype=jnp.float32)
  true_fwd_hiddens_1 = jnp.array([0., 0., 0.125, 0.005859375], dtype=jnp.float32)
  true_bwd_cells_1 = jnp.array([0., 0.046875, 0.25, 0.], dtype=jnp.float32)
  true_bwd_hiddens_1 = jnp.array([0., 0.005859375, 0.125, 0.], dtype=jnp.float32)
  assert jnp.allclose(fwd_state_1.hidden, true_fwd_hiddens_1.reshape(fwd_state_1.hidden.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(fwd_state_1.cell, true_fwd_cells_1.reshape(fwd_state_1.hidden.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_1.hidden, true_bwd_hiddens_1.reshape(bwd_state_1.cell.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_1.cell, true_bwd_cells_1.reshape(bwd_state_1.cell.shape), rtol=1e-20, atol=1e-8)


  fwd_state_2, bwd_state_2 = states[1]
  true_fwd_cells_2 = jnp.array([0., 0.00003433227539, 0.06250868366, 0.001804768683], dtype=jnp.float32)
  true_fwd_hiddens_2 = jnp.array([0., 0.0000002011656761, 0.01562718349, 0.00003877826787], dtype=jnp.float32)
  true_bwd_cells_2 = jnp.array([0.0000000714895575, 0.001804768683, 0.06250868366, 0.00003433227539], dtype=jnp.float32)
  true_bwd_hiddens_2 = jnp.array([2.772241211e-12, 0.00003877826787, 0.01562718349, 0.0000002011656761], dtype=jnp.float32)

  assert jnp.allclose(fwd_state_2.hidden, true_fwd_hiddens_2.reshape(fwd_state_2.hidden.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(fwd_state_2.cell, true_fwd_cells_2.reshape(fwd_state_2.hidden.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_2.hidden, true_bwd_hiddens_2.reshape(bwd_state_2.cell.shape), rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_2.cell, true_bwd_cells_2.reshape(bwd_state_2.cell.shape), rtol=1e-20, atol=1e-8)

  true_out = jnp.stack([true_fwd_hiddens_2, true_bwd_hiddens_2], axis=1)
  assert jnp.allclose(outs, true_out, rtol=1e-20, atol=1e-8)


def test_deep_identity_bilstm_not_full_length():
  input_size=1
  max_seq_len=4
  length=3

  key = jax.random.PRNGKey(0)
  rnn = recurrent.BiLSTM(
          key,
          input_size,
          [input_size, input_size],
          act_fn=jax.nn.relu,
          recurrent_act_fn=jax.nn.relu,
          input_W_init=jax.nn.initializers.ones,
          recurrent_W_init=jax.nn.initializers.ones,
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  inputs = jnp.array([-1., 0., .5, 10.]).reshape([max_seq_len, input_size])
  states, outs = rnn(inputs, length=length, initial_state=None)
  fwd_state_1, bwd_state_1 = states[0]
  true_fwd_cells_1 = jnp.array([0., 0., 0.25], dtype=jnp.float32)[:, jnp.newaxis]
  true_fwd_hiddens_1 = jnp.array([0., 0., 0.125], dtype=jnp.float32)[:, jnp.newaxis]
  true_bwd_cells_1 = jnp.array([0., 0.046875, 0.25], dtype=jnp.float32)[:, jnp.newaxis]
  true_bwd_hiddens_1 = jnp.array([0., 0.005859375, 0.125], dtype=jnp.float32)[:, jnp.newaxis]
  assert jnp.allclose(fwd_state_1.hidden[:length], true_fwd_hiddens_1, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(fwd_state_1.cell[:length], true_fwd_cells_1, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_1.hidden[:length], true_bwd_hiddens_1, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_1.cell[:length], true_bwd_cells_1, rtol=1e-20, atol=1e-8)


  fwd_state_2, bwd_state_2 = states[1]
  true_fwd_cells_2 = jnp.array([0., 0.00003433227539, 0.06250868366], dtype=jnp.float32)[:, jnp.newaxis]
  true_fwd_hiddens_2 = jnp.array([0., 0.0000002011656761, 0.01562718349], dtype=jnp.float32)[:, jnp.newaxis]
  true_bwd_cells_2 = jnp.array([0.0000000714491209, 0.001804351807, 0.0625], dtype=jnp.float32)[:, jnp.newaxis]
  true_bwd_hiddens_2 = jnp.array([2.769751669e-12, 0.00003876537085, 0.015625], dtype=jnp.float32)[:, jnp.newaxis]

  assert jnp.allclose(fwd_state_2.hidden[:length], true_fwd_hiddens_2, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(fwd_state_2.cell[:length], true_fwd_cells_2, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_2.hidden[:length], true_bwd_hiddens_2, rtol=1e-20, atol=1e-8)
  assert jnp.allclose(bwd_state_2.cell[:length], true_bwd_cells_2, rtol=1e-20, atol=1e-8)

  true_out = jnp.concatenate([true_fwd_hiddens_2, true_bwd_hiddens_2], axis=1)
  assert jnp.allclose(outs[:length], true_out, rtol=1e-20, atol=1e-8)


class SimpleCellState(eqx.Module):
  value: Array

class SimpleCell(eqx.Module):

  hidden_dim: int = eqx.static_field()
  out_dim: int = eqx.static_field()

  def __init__(self):
    self.hidden_dim = 1
    self.out_dim = 1

  def __call__(self,
               prev_state: SimpleCellState,
               inputs: Array
               ) -> Tuple[SimpleCellState, Array]:
    new_state = prev_state.value + inputs
    return SimpleCellState(value=new_state), new_state

  def initial_state(self) -> SimpleCellState:
    return SimpleCellState(value=jnp.array([0]))


class SimpleRNN(recurrent.RNN[SimpleCellState]):

  def __init__(self):
    super().__init__(jax.random.PRNGKey(0), 1, [1], lambda *_: SimpleCell())


class SimpleBiRNN(recurrent.BiRNN[SimpleCellState]):

  def __init__(self):
    super().__init__(jax.random.PRNGKey(0), 1, [1], lambda *_: SimpleCell())


def test_simple_rnn():
  rnn = SimpleRNN()
  inputs = jnp.arange(10)
  state_check = jnp.cumsum(inputs, axis=0).reshape([10,1])
  for l in range(1, 10):
    state, out = rnn(inputs, length=l)
    assert jnp.allclose(state[0].value, state_check)
    assert jnp.allclose(out, state_check)

def test_simple_rnn_reverse():
  rnn = SimpleRNN()
  inputs = jnp.arange(10)
  for l in range(1, 10):
    state, out = rnn(inputs, length=l, reverse=True)
    state_check = jnp.cumsum(inputs[:l][::-1])[::-1].reshape([l,1])
    assert jnp.allclose(state[0].value[:l], state_check)
    assert jnp.allclose(out[:l], state_check)

  # no length provided
  state, out = rnn(inputs, reverse=True)
  state_check = jnp.cumsum(inputs[::-1])[::-1].reshape([10,1])
  assert jnp.allclose(state[0].value, state_check)
  assert jnp.allclose(out, state_check)

def test_simple_birnn():
  rnn = SimpleBiRNN()
  inputs = jnp.arange(10).reshape([10, 1])
  for l in range(1, 10):
    states, out = rnn(inputs, length=l)
    fwd_state, bwd_state = states[0]
    bwd_state_check = jnp.cumsum(inputs[:l][::-1])[::-1].reshape([l,1])
    fwd_state_check = jnp.cumsum(inputs[:l]).reshape([l,1])
    assert jnp.allclose(fwd_state.value[:l], fwd_state_check)
    assert jnp.allclose(bwd_state.value[:l], bwd_state_check)
    assert jnp.allclose(out[:l, 0], fwd_state_check.squeeze())
    assert jnp.allclose(out[:l, 1], bwd_state_check.squeeze())

  # no length provided
  states, out = rnn(inputs)
  fwd_state, bwd_state = states[0]
  bwd_state_check = jnp.cumsum(inputs[::-1])[::-1][:, jnp.newaxis]
  fwd_state_check = jnp.cumsum(inputs)[:, jnp.newaxis]
  assert jnp.allclose(fwd_state.value, fwd_state_check)
  assert jnp.allclose(bwd_state.value, bwd_state_check)
  assert jnp.allclose(out[:, 0], fwd_state_check.squeeze())
  assert jnp.allclose(out[:, 1], bwd_state_check.squeeze())

def test_flatten():
  key = jax.random.PRNGKey(0)
  rnn = recurrent.VanillaRNN(key, 2, [4])
  leaves, _ = jax.tree_util.tree_flatten(rnn)
  assert len(leaves) == 3
  for l in leaves:
    assert l.dtype == jnp.float32
