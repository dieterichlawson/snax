import os
import tempfile
import jax
import jax.numpy as jnp
import optax
from . import checkpoint
from . import recurrent
from . import train_lib

def test_checkpoint_dir_not_exists():
  data = jax.random.uniform(jax.random.PRNGKey(0), shape=[10,10])
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    new_dir = os.path.join(dirname, "tmp")
    checkpoint.save_checkpoint(data, 1, new_dir)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(data, new_dir)
    assert jnp.allclose(data, reloaded_data)
    assert step == reloaded_step

def test_checkpoint():
  data = jax.random.uniform(jax.random.PRNGKey(0), shape=[10,10])
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(data, 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(data, dirname)
    assert jnp.allclose(data, reloaded_data)
    assert step == reloaded_step

def test_checkpoint_rnn():
  model = recurrent.LSTMCell(jax.random.PRNGKey(0), 2, 3, forget_gate_bias_init=12.)
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(model, 1, dirname)
    reloaded_data, reloaded_step =  checkpoint.load_latest_checkpoint(model, dirname)
    assert reloaded_data is not None, "Checkpoint loading failed."
    def assert_close(x, y):
      assert jnp.allclose(x, y), "%s, %s" % (str(x), str(y))

    # Check that the parameters are the same
    flat_model, _ = jax.tree_util.tree_flatten(model)
    flat_reloaded, _ = jax.tree_util.tree_flatten(reloaded_data)
    [assert_close(x, y) for x,y in zip(flat_model, flat_reloaded)]
    # Check the step is the same
    assert step == reloaded_step
    initial_state = model.initial_state()
    # Check that the reloaded checkpoint computes the same function
    model_outs, _ = jax.tree_util.tree_flatten(model(initial_state, jnp.ones(2)))
    check_outs, _ = jax.tree_util.tree_flatten(reloaded_data(initial_state, jnp.ones(2)))
    [assert_close(x, y) for x, y in zip(model_outs, check_outs)]


def test_checkpoint_rnn2():
  model = recurrent.LSTMCell(jax.random.PRNGKey(0), 2, 3, forget_gate_bias_init=12.)
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(model, 1, dirname)
    reloaded_data, reloaded_step =  checkpoint.load_latest_checkpoint(model, dirname)

    assert reloaded_data is not None, "Checkpoint loading failed."
    def assert_close(x, y):
      assert jnp.allclose(x, y), "%s, %s" % (str(x), str(y))

    # Check that the parameters are the same
    flat_model, _ = jax.tree_util.tree_flatten(model)
    flat_reloaded, _ = jax.tree_util.tree_flatten(reloaded_data)
    [assert_close(x, y) for x,y in zip(flat_model, flat_reloaded)]
    # Check the step is the same
    assert step == reloaded_step
    initial_state = model.initial_state()
    # Check that the reloaded checkpoint computes the same function
    model_outs, _ = jax.tree_util.tree_flatten(model(initial_state, jnp.ones(2)))
    check_outs, _ = jax.tree_util.tree_flatten(reloaded_data(initial_state, jnp.ones(2)))
    [assert_close(x, y) for x, y in zip(model_outs, check_outs)]

    # Check that you can compose and re-jit those functions
    @jax.jit
    def f(x):
      state, out = reloaded_data(reloaded_data.initial_state(), jnp.ones(2))
      return out + 3.

    @jax.jit
    def g(x):
      state, out = model(model.initial_state(), jnp.ones(2))
      return out + 3.

    model_outs, _ = jax.tree_util.tree_flatten(g(jnp.ones(2)))
    check_outs, _ = jax.tree_util.tree_flatten(f(jnp.ones(2)))
    [assert_close(x, y) for x, y in zip(model_outs, check_outs)]


def test_checkpoint_multi():
  datas = [jax.random.uniform(jax.random.PRNGKey(i), shape=[10,10]) for i in range(3)]
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(datas[0], 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(datas[0], dirname)
    assert jnp.allclose(datas[0], reloaded_data)
    assert 1 == reloaded_step
    checkpoint.save_checkpoint(datas[1], 2, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(datas[1], dirname)
    assert jnp.allclose(datas[1], reloaded_data)
    assert 2 == reloaded_step
    checkpoint.save_checkpoint(datas[2], 3, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(datas[2], dirname)
    assert jnp.allclose(datas[2], reloaded_data)
    assert 3 == reloaded_step

def test_removing_checkpoint():
  datas = [jax.random.uniform(jax.random.PRNGKey(i), shape=[10,10]) for i in range(5)]
  with tempfile.TemporaryDirectory() as dirname:
    for data, step in zip(datas, range(5)):
      checkpoint.save_checkpoint(data, step + 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(datas[-1], dirname)
    assert jnp.allclose(datas[4], reloaded_data)
    assert 5 == reloaded_step
    steps = [checkpoint.step_from_path(x) for x in checkpoint.get_checkpoints(dirname)]
    assert 1 not in steps
    assert 2 not in steps
    assert 3 in steps
    assert 4 in steps
    assert 5 in steps


def test_load_checkpoint_with_treedef():
  data = jax.random.uniform(jax.random.PRNGKey(0), shape=[10,10])
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(data, 1, dirname)
    reloaded_data, _ = checkpoint.load_latest_checkpoint(data, dirname)
    assert reloaded_data is not None

    def for_body(i, val):
      return val + reloaded_data[i]

    reloaded_sum = jax.lax.fori_loop(0, 10, for_body, jnp.zeros([10]))
    true_sum = jnp.sum(data, axis=0)
    assert jnp.allclose(reloaded_sum, true_sum)


def test_model_only_checkpoint():

  def loss_fn(key, step, params):
    return jnp.mean(jnp.square(params))

  opt = optax.adam(1e-2)
  init_params = jnp.array(1.)

  with tempfile.TemporaryDirectory() as dirname:
    out = train_lib.train(
            jax.random.PRNGKey(0),
            loss_fn,
            opt,
            init_params,
            parallelize=False,
            batch_size=16,
            num_steps=10,
            summarize_every=5,
            checkpoint_every=5,
            checkpoints_to_keep=3,
            checkpoint_dir=dirname)
    import IPython
    IPython.embed()

    reloaded_data, step = checkpoint.load_latest_checkpoint_model_only(
            init_params, dirname)
    assert step == 10

    assert reloaded_data is not None, "Checkpoint loading failed."
    assert jnp.allclose(out, reloaded_data)
