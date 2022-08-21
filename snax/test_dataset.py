import jax
import jax.numpy as jnp
from .dataset import InMemDataset
import pytest
import math

EVEN_SIZES = [(1, 100), (10, 100), (50, 100), (100, 100)]
UNEVEN_SIZES = [(3, 100), (27, 100), (99, 100)]
BIGGER_SIZES = [(101, 100), (120, 100), (200, 100)]
ALL_SIZES = EVEN_SIZES + UNEVEN_SIZES + BIGGER_SIZES

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_dataset_sees_all_data(batch_size, dataset_size, shuffle):
  data = jnp.arange(1, dataset_size, dtype=jnp.int32)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  tot = ds.batch_sum_reduce(lambda x: jnp.sum(x))
  assert jnp.equal(tot, jnp.sum(data))

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_dataset_sees_all_data2(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def while_pred(state):
    _, _, _, lb = state
    return jnp.logical_not(lb)

  def while_body(state):
    data, ds, i, _ = state
    batch, mask, lb, new_ds = ds.next()
    to_write = jnp.where(
            mask,
            batch,
            jnp.full_like(batch, jnp.nan))
    new_data = data.at[i].set(to_write)
    return new_data, new_ds, i+1, lb

  num_batches = math.ceil(dataset_size / batch_size)
  init_data = jnp.zeros([num_batches, batch_size], dtype = jnp.float32)

  outs = jax.lax.while_loop(while_pred, while_body, (init_data, ds, 0, False))

  seen_data = outs[0]
  d = seen_data.reshape([-1])
  seen_set = set()
  data_set = set([float(x) for x in data])
  for x in d:
    if not jnp.isnan(x):
      assert float(x) not in seen_set
      seen_set.add(float(x))

  assert data_set == seen_set


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", EVEN_SIZES)
def test_mask_correct_batch_even(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def scan_body(d, _):
    _, mask, _, new_d = d.next()
    return new_d, mask

  num_batches = dataset_size // batch_size

  _, outs = jax.lax.scan(scan_body, ds, jnp.arange(num_batches))

  assert jnp.all(jnp.equal(outs, jnp.ones_like(outs)))

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", UNEVEN_SIZES + BIGGER_SIZES)
def test_mask_correct_batch_uneven(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def scan_body(d, _):
    _, mask, _, new_d = d.next()
    return new_d, mask

  num_batches = math.ceil(dataset_size / batch_size)

  _, outs = jax.lax.scan(scan_body, ds, jnp.arange(num_batches))

  out_masks = outs.reshape([-1])
  true_mask = jnp.array(
          1. * (jnp.arange(num_batches * batch_size) < dataset_size),
          dtype=jnp.float32)
  assert jnp.all(out_masks == true_mask)

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size",
        [(1, 100), (3, 100), (50, 100), (100, 100), (120, 100), (20, 100), (13, 100), (200, 100)])
def test_lb_correct(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def scan_body(ds, _):
    _, _, lb, new_ds = ds.next()
    return new_ds, lb

  num_batches = math.ceil(dataset_size / batch_size)
  _, lbs = jax.lax.scan(scan_body, ds, jnp.arange(10 * num_batches))

  true_lbs = jnp.arange(1, (10 * num_batches) + 1) % num_batches == 0
  assert jnp.all(lbs == true_lbs)

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size",
        [(1, 100), (3, 100), (50, 100), (100, 100), (120, 100), (20, 100), (13, 100), (200, 100)])
def test_arraytree(batch_size, dataset_size, shuffle):
  data1 = jnp.arange(dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(dataset_size, dtype=jnp.float32)

  data = ((data1, data2, data3) , data4)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def reduce_f(x):
    (a, b, c), d = x
    return a * b * c * d

  out = ds.batch_sum_reduce(reduce_f)
  true_out = jnp.sum(data1 * data2 * data3 * data4)
  assert jnp.allclose(out, true_out)

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_arraytree_init_acc(batch_size, dataset_size, shuffle):
  data1 = jnp.arange(dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(dataset_size, dtype=jnp.float32)

  data = ((data1, data2, data3) , data4)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def reduce_f(x):
    (a, b, c), d = x
    return (a * b), 2 * (c * d)

  out = ds.batch_sum_reduce(reduce_f)
  true_out = (jnp.sum(data1 * data2), jnp.sum(2 * (data3 * data4)))
  assert jnp.allclose(out[0], true_out[0]) and jnp.allclose(out[1], true_out[1])


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_arraytree_init_acc_multi_d(batch_size, dataset_size, shuffle):
  data1 = jnp.arange(2*dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(2*dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(2*dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(2*dataset_size, dtype=jnp.float32)

  data1 = data1.reshape([dataset_size, 2])
  data2 = data2.reshape([dataset_size, 2])
  data3 = data3.reshape([dataset_size, 2])
  data4 = data4.reshape([dataset_size, 2])

  data = ((data1, data2, data3) , data4)
  key = jax.random.PRNGKey(0)
  ds = InMemDataset(data, batch_size, shuffle=shuffle, key=key)

  def reduce_f(x):
    (a, b, c), d = x
    return (a * b), 2 * (c * d)

  out = ds.batch_sum_reduce(reduce_f)
  true_out = (jnp.sum(data1 * data2, axis=0), jnp.sum(2 * (data3 * data4), axis=0))
  assert jnp.allclose(out[0], true_out[0]) and jnp.allclose(out[1], true_out[1])
