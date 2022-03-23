import os
import tempfile
import jax
import jax.numpy as jnp
from . import checkpoint

def test_checkpoint_dir_not_exists():
  data = jax.random.uniform(jax.random.PRNGKey(0), shape=[10,10])
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    new_dir = os.path.join(dirname, "tmp")
    checkpoint.save_checkpoint(data, 1, new_dir)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(new_dir)
    assert jnp.allclose(data, reloaded_data)
    assert step == reloaded_step

def test_checkpoint():
  data = jax.random.uniform(jax.random.PRNGKey(0), shape=[10,10])
  step = 1
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(data, 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(dirname)
    assert jnp.allclose(data, reloaded_data)
    assert step == reloaded_step

def test_checkpoint_multi():
  datas = [jax.random.uniform(jax.random.PRNGKey(i), shape=[10,10]) for i in range(3)]
  with tempfile.TemporaryDirectory() as dirname:
    checkpoint.save_checkpoint(datas[0], 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(dirname)
    assert jnp.allclose(datas[0], reloaded_data)
    assert 1 == reloaded_step
    checkpoint.save_checkpoint(datas[1], 2, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(dirname)
    assert jnp.allclose(datas[1], reloaded_data)
    assert 2 == reloaded_step
    checkpoint.save_checkpoint(datas[2], 3, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(dirname)
    assert jnp.allclose(datas[2], reloaded_data)
    assert 3 == reloaded_step

def test_removing_checkpoint():
  datas = [jax.random.uniform(jax.random.PRNGKey(i), shape=[10,10]) for i in range(5)]
  with tempfile.TemporaryDirectory() as dirname:
    for data, step in zip(datas, range(5)):
      checkpoint.save_checkpoint(data, step + 1, dirname)
    reloaded_data, reloaded_step = checkpoint.load_latest_checkpoint(dirname)
    assert jnp.allclose(datas[4], reloaded_data)
    assert 5 == reloaded_step
    steps = [checkpoint.step_from_path(x) for x in checkpoint.get_checkpoints(dirname)]
    assert 1 not in steps
    assert 2 not in steps
    assert 3 in steps
    assert 4 in steps
    assert 5 in steps
