import jax
import jax.numpy as jnp

from . import recurrent
from . import utils

def test_basic_vanilla_rnn(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  key = jax.random.PRNGKey(0)
  rnn = recurrent.VanillaRNN(key, input_size, hidden_sizes)
  inputs = jnp.ones([seq_len, input_size])

  def call(model, inputs):
    return model(inputs, None)

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
    return model(inputs, None)

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
    return model(inputs, None)

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
    return model(inputs, None)

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
  states, out = rnn(inputs, None)
  state_check = inputs
  for state in states:
    state_check = jnp.cumsum(state_check, axis=0)
    assert jnp.allclose(state.hidden, state_check)
  assert jnp.allclose(out, state_check)

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
  states, _ = rnn(inputs, None)
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
  inputs = jnp.array([1., 0., 0., -1.]).reshape([seq_len, input_size])
  states, outs = rnn(inputs, seq_len, None)
  fwd_state, bwd_state = states[0]
  true_fwd_hiddens = jnp.array([1., 2., 16., (15**2)*(8+15)])
  true_fwd_cells = jnp.array([1., 2., 8., 15*(8+15)])
  true_bwd_hiddens = jnp.array([0., 0., 0., 1.])
  true_bwd_cells = jnp.array([0., 0., 0., 1.])
  assert jnp.allclose(fwd_state.hidden, true_fwd_hiddens.reshape(fwd_state.hidden.shape))
  assert jnp.allclose(fwd_state.cell, true_fwd_cells.reshape(fwd_state.hidden.shape))
  assert jnp.allclose(bwd_state.hidden, true_bwd_hiddens.reshape(bwd_state.cell.shape))
  assert jnp.allclose(bwd_state.cell, true_bwd_cells.reshape(bwd_state.cell.shape))
  true_out = jnp.stack([true_fwd_hiddens, true_bwd_hiddens], axis=1)
  assert jnp.allclose(outs, true_out.reshape(outs.shape))
