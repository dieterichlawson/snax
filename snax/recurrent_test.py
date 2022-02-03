import recurrent_protocol
import jax
import jax.numpy as jnp
import utils

def test_basic_vanilla_rnn(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.VanillaRNN(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)

def test_basic_vanilla_birnn(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.BidirectionalVanillaRNN(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
  assert out.shape == (seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (seq_len, size)
    assert bwd_state.hidden.shape == (seq_len, size)

def test_basic_lstm(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.LSTM(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)
    assert state.cell.shape == (seq_len, size)

def test_basic_bilstm(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.BidirectionalLSTM(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
  assert out.shape == (seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (seq_len, size)
    assert fwd_state.cell.shape == (seq_len, size)
    assert bwd_state.hidden.shape == (seq_len, size)
    assert bwd_state.cell.shape == (seq_len, size)

def test_basic_gru(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.GRU(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
  assert out.shape == (seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (seq_len, size)

def test_basic_bigru(hidden_sizes=[3,4,5], input_size=2, seq_len=11):
  rnn = recurrent_protocol.BidirectionalGRU(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.ones([seq_len, input_size])
  states, out = jax.jit(rnn.apply)(params, inputs, seq_len, None)
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
  inputs = jnp.ones([batch_size, seq_len, input_size])
  lengths =  jnp.array(lengths)

  rnn = recurrent_protocol.GRU(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  states, out = jax.vmap(rnn.apply, in_axes=(None, 0, 0, None))(params, inputs, lengths, None)
  assert out.shape == (batch_size, seq_len, hidden_sizes[-1])
  for size, state in zip(hidden_sizes, states):
    assert state.hidden.shape == (batch_size, seq_len, size)

def test_batch_bigru(
        batch_size=4,
        hidden_sizes=[3,4,5],
        input_size=2,
        seq_len=11,
        lengths=[1,11,0,9]):
  inputs = jnp.ones([batch_size, seq_len, input_size])
  lengths =  jnp.array(lengths)

  rnn = recurrent_protocol.BidirectionalGRU(hidden_sizes)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  states, out = jax.vmap(rnn.apply, in_axes=(None, 0, 0, None))(params, inputs, lengths, None)
  assert out.shape == (batch_size, seq_len, hidden_sizes[-1]*2)
  for size, state in zip(hidden_sizes, states):
    fwd_state, bwd_state = state
    assert fwd_state.hidden.shape == (batch_size, seq_len, size)
    assert bwd_state.hidden.shape == (batch_size, seq_len, size)

def test_identity_vanilla_rnn(num_layers=3, input_size=2, seq_len=5):
  rnn = recurrent_protocol.VanillaRNN(
          [input_size] * num_layers,
          activation=jax.nn.relu,
          input_W_init=utils.identity_init,
          recurrent_W_init=utils.identity_init,
          b_init=jax.nn.initializers.zeros)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.arange(seq_len * input_size).reshape([seq_len, input_size])
  states, out = rnn.apply(params, inputs, seq_len, None)
  state_check = inputs
  for state in states:
    state_check = jnp.cumsum(state_check, axis=0)
    assert jnp.allclose(state.hidden, state_check)
  assert jnp.allclose(out, state_check)

def test_identity_lstmcell():
  input_size=1
  cell = recurrent_protocol.LSTMCell(
          input_size,
          activation=jax.nn.relu,
          recurrent_activation=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  _, params = cell.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.array([2.])
  state, _ = cell.apply(params, inputs, cell.initial_state())
  assert jnp.allclose(state.hidden, jnp.array([2**3]))
  assert jnp.allclose(state.cell, jnp.array([2**2]))

def test_identity_lstm():
  input_size=1
  seq_len=4
  rnn = recurrent_protocol.LSTM(
          [input_size],
          activation=jax.nn.relu,
          recurrent_activation=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.arange(seq_len * input_size).reshape([seq_len, input_size])
  states, _ = rnn.apply(params, inputs, seq_len, None)
  states = states[0]
  true_hiddens = jnp.array([0, 1, 36, 39*(39*12 + 39**2)], dtype=jnp.float32)
  true_cells = jnp.array([0, 1, 12, 39*12 + 39**2], dtype=jnp.float32)
  assert jnp.allclose(states.hidden, true_hiddens.reshape(states.hidden.shape))
  assert jnp.allclose(states.cell, true_cells.reshape(states.cell.shape))

def test_identity_bilstm():
  input_size=1
  seq_len=4
  rnn = recurrent_protocol.BidirectionalLSTM(
          [input_size],
          activation=jax.nn.relu,
          recurrent_activation=jax.nn.relu,
          input_W_init=utils.stack_identity_init(4),
          recurrent_W_init=utils.stack_identity_init(4),
          b_init=jax.nn.initializers.zeros,
          forget_gate_bias_init=0.)
  _, params = rnn.init(jax.random.PRNGKey(0), input_size)
  inputs = jnp.array([1., 0., 0., -1.]).reshape([seq_len, input_size])
  states, outs = rnn.apply(params, inputs, seq_len, None)
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

test_basic_vanilla_rnn()
test_basic_vanilla_birnn()
test_basic_lstm()
test_basic_bilstm()
test_basic_gru()
test_basic_bigru()
test_batch_gru()
test_batch_bigru()
test_identity_vanilla_rnn()
test_identity_lstm()
test_identity_lstmcell()
test_identity_bilstm()
