import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import tempfile
import jax
import jax.numpy as jnp
from . import train_lib
from . import dataset
import optax
import pytest


def test_nan_break_with_checkpoint():

  def loss_fn(key, step, params):
    return jnp.mean(jnp.power(params, params))

  opt = optax.adam(10.)
  with tempfile.TemporaryDirectory() as tmp_dir:
    with pytest.raises(ValueError):
      out = train_lib.train(
            jax.random.PRNGKey(0),
            loss_fn,
            opt,
            jnp.array(10.),
            parallelize=False,
            batch_size=16,
            num_steps=1000,
            break_on_nan=True,
            checkpoint_dir=tmp_dir)
    assert "nan_checkpoint_00000003.chk" in os.listdir(tmp_dir)

def test_nan_break():

  def loss_fn(key, step, params):
    return jnp.mean(jnp.power(params, params))

  opt = optax.adam(10.)
  with pytest.raises(ValueError):
    out = train_lib.train(
          jax.random.PRNGKey(0),
          loss_fn,
          opt,
          jnp.array(10.),
          parallelize=False,
          batch_size=16,
          num_steps=1000,
          break_on_nan=True)

def test_train_no_dataset_no_parallel():

  def loss_fn(key, step, params):
    return jnp.mean(jnp.square(params))

  opt = optax.adam(1e-2)

  out = train_lib.train(
          jax.random.PRNGKey(0),
          loss_fn,
          opt,
          jnp.array(10.),
          parallelize=False,
          batch_size=16,
          num_steps=1000)

def test_train_no_dataset_parallel():

  def loss_fn(key, step, params):
    return jnp.mean(jnp.square(params))

  opt = optax.adam(1e-2)

  out = train_lib.train(
          jax.random.PRNGKey(0),
          loss_fn,
          opt,
          jnp.array(10.),
          parallelize=True,
          batch_size=16,
          num_steps=1000)

def test_train_with_dataset_no_parallelize():
  data = jnp.arange(1000, dtype=jnp.float32)
  ds = dataset.InMemDataset(data, 16, shuffle=True)

  def loss_fn(key, step, params, d):
    return jnp.mean(jnp.square(params - d))

  opt = optax.adam(1e-2)

  out = train_lib.train(
          jax.random.PRNGKey(0),
          loss_fn,
          opt,
          jnp.array(10.),
          dataset=ds,
          parallelize=False,
          batch_size=16,
          num_steps=1000)

def test_train_with_dataset_parallelize():
  data = jnp.arange(1000, dtype=jnp.float32)
  ds = dataset.InMemDataset(data, 4, shuffle=True)

  def loss_fn(key, step, params, d):
    return jnp.mean(jnp.square(params - d))

  opt = optax.adam(1e-2)

  out = train_lib.train(
          jax.random.PRNGKey(0),
          loss_fn,
          opt,
          jnp.array(10.),
          dataset=ds,
          parallelize=True,
          batch_size=16,
          num_steps=1000)

@pytest.mark.parametrize("num_inner_steps", [1, 100])
def test_interleaved(num_inner_steps):
  data = jnp.arange(1000, dtype=jnp.float32)
  ds = dataset.InMemDataset(data, 4, shuffle=True)

  def loss_fn1(key, step, params, d):
    return jnp.mean(jnp.square(params - d))

  opt = optax.adam(1e-2)

  ts1 = train_lib.TrainStep(loss_fn1, opt,
          dataset=ds, batch_size=16, parallelize=True, num_inner_steps=num_inner_steps,
          name="loss1")

  def loss_fn2(key, step, params, d):
    return jnp.mean(jnp.square(params - d*2))

  opt = optax.adam(1e-2)

  ts2 = train_lib.TrainStep(loss_fn2, opt,
          dataset=ds, batch_size=4, parallelize=False, num_inner_steps=num_inner_steps,
          name="loss2")

  out = train_lib.train_alternating(
          jax.random.PRNGKey(0),
          [ts1, ts2],
          jnp.array(10.),
          num_steps=1000)
