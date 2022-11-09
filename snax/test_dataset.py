import jax
import jax.numpy as jnp
import numpy as onp
from .dataset import InMemDataset
import pytest
import math

EVEN_SIZES = [(1, 100), (10, 100), (50, 100), (100, 100)]
UNEVEN_SIZES = [(3, 100), (27, 100), (99, 100)]
BIGGER_SIZES = [(101, 100), (120, 100), (200, 100)]
ALL_SIZES = EVEN_SIZES + UNEVEN_SIZES + BIGGER_SIZES

@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_works_with_ordinary_numpy(batch_size, dataset_size):
  data = onp.arange(1, dataset_size, dtype=jnp.int32)
  ds = InMemDataset(data, batch_size)
  tot = ds.batch_sum_reduce(lambda x: jnp.sum(x))
  assert jnp.equal(tot, jnp.sum(data))


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_dataset_sees_all_data(batch_size, dataset_size):
  data = jnp.arange(1, dataset_size, dtype=jnp.int32)
  ds = InMemDataset(data, batch_size)

  tot = ds.batch_sum_reduce(lambda x: jnp.sum(x))
  assert jnp.equal(tot, jnp.sum(data))


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_dataset_sees_all_data2(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  ds = InMemDataset(data, batch_size, shuffle=shuffle)

  def while_pred(state):
    _, _, _, lb = state
    return jnp.logical_not(lb)

  def while_body(state):
    data, s, i, _ = state
    batch, mask, lb, new_s = ds.next(s)
    to_write = jnp.where(
            mask,
            batch,
            jnp.full_like(batch, jnp.nan))
    new_data = data.at[i].set(to_write)
    return new_data, new_s, i+1, lb

  num_batches = math.ceil(dataset_size / batch_size)
  init_data = jnp.zeros([num_batches, batch_size], dtype = jnp.float32)
  key = jax.random.PRNGKey(0)
  init_s = ds.init_state(key)
  outs = jax.lax.while_loop(while_pred, while_body, (init_data, init_s, 0, False))

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
  ds = InMemDataset(data, batch_size, shuffle=shuffle)

  def scan_body(s, _):
    _, mask, _, new_s = ds.next(s)
    return new_s, mask

  num_batches = dataset_size // batch_size
  key = jax.random.PRNGKey(0)
  init_s = ds.init_state(key)
  _, outs = jax.lax.scan(scan_body, init_s, jnp.arange(num_batches))

  assert jnp.all(jnp.equal(outs, jnp.ones_like(outs)))


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", UNEVEN_SIZES + BIGGER_SIZES)
def test_mask_correct_batch_uneven(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  ds = InMemDataset(data, batch_size, shuffle=shuffle)

  def scan_body(s, _):
    _, mask, _, new_s = ds.next(s)
    return new_s, mask

  num_batches = math.ceil(dataset_size / batch_size)
  key = jax.random.PRNGKey(0)
  init_s = ds.init_state(key)
  _, outs = jax.lax.scan(scan_body, init_s, jnp.arange(num_batches))

  out_masks = outs.reshape([-1])
  true_mask = jnp.array(
          1. * (jnp.arange(num_batches * batch_size) < dataset_size),
          dtype=jnp.float32)
  assert jnp.all(out_masks == true_mask)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_lb_correct(batch_size, dataset_size, shuffle):
  data = jnp.arange(dataset_size, dtype=jnp.float32)
  ds = InMemDataset(data, batch_size, shuffle=shuffle)

  def scan_body(s, _):
    _, _, lb, new_s = ds.next(s)
    return new_s, lb

  num_batches = math.ceil(dataset_size / batch_size)
  key = jax.random.PRNGKey(0)
  init_s = ds.init_state(key)
  _, lbs = jax.lax.scan(scan_body, init_s, jnp.arange(10 * num_batches))

  true_lbs = jnp.arange(1, (10 * num_batches) + 1) % num_batches == 0
  assert jnp.all(lbs == true_lbs)


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_arraytree(batch_size, dataset_size):
  data1 = jnp.arange(dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(dataset_size, dtype=jnp.float32)

  data = ((data1, data2, data3) , data4)
  ds = InMemDataset(data, batch_size)

  def reduce_f(x):
    (a, b, c), d = x
    return a * b * c * d

  out = ds.batch_sum_reduce(reduce_f)
  true_out = jnp.sum(data1 * data2 * data3 * data4)
  assert jnp.allclose(out, true_out)


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_arraytree_init_acc(batch_size, dataset_size):
  data1 = jnp.arange(dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(dataset_size, dtype=jnp.float32)

  data = ((data1, data2, data3) , data4)
  ds = InMemDataset(data, batch_size)

  def reduce_f(x):
    (a, b, c), d = x
    return (a * b), 2 * (c * d)

  out = ds.batch_sum_reduce(reduce_f)
  true_out = (jnp.sum(data1 * data2), jnp.sum(2 * (data3 * data4)))
  assert jnp.allclose(out[0], true_out[0]) and jnp.allclose(out[1], true_out[1])


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_arraytree_init_acc_multi_d(batch_size, dataset_size):
  data1 = jnp.arange(2*dataset_size, dtype=jnp.float32)
  data2 = 2 * jnp.arange(2*dataset_size, dtype=jnp.float32)
  data3 = 3 * jnp.arange(2*dataset_size, dtype=jnp.float32)
  data4 = 4 * jnp.arange(2*dataset_size, dtype=jnp.float32)

  data1 = data1.reshape([dataset_size, 2])
  data2 = data2.reshape([dataset_size, 2])
  data3 = data3.reshape([dataset_size, 2])
  data4 = data4.reshape([dataset_size, 2])

  data = ((data1, data2, data3) , data4)
  ds = InMemDataset(data, batch_size)

  def reduce_f(x):
    (a, b, c), d = x
    return (a * b), 2 * (c * d)

  out = ds.batch_sum_reduce(reduce_f)
  true_out = (jnp.sum(data1 * data2, axis=0), jnp.sum(2 * (data3 * data4), axis=0))
  assert jnp.allclose(out[0], true_out[0]) and jnp.allclose(out[1], true_out[1])


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_pmap(monkeypatch, batch_size, dataset_size):
  with monkeypatch.context() as m:
    m.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4", prepend=',')
    data = jnp.arange(dataset_size, dtype=jnp.float32)
    ds = InMemDataset(data, batch_size, shuffle=True)
    key = jax.random.PRNGKey(0)
    keys = [jax.random.fold_in(key, i) for i in range(jax.local_device_count())]
    list_init_s = [ds.init_state(k) for k in keys]
    shd_init_s = jax.device_put_sharded(list_init_s, jax.local_devices())

    def while_pred(state):
      _, _, _, lb = state
      return jnp.logical_not(lb)

    def while_body(state):
      data, s, i, _ = state
      batch, mask, lb, new_s = ds.next(s)
      to_write = jnp.where(
              mask,
              batch,
              jnp.full_like(batch, jnp.nan))
      new_data = data.at[i].set(to_write)
      return new_data, new_s, i+1, lb

    num_batches = math.ceil(dataset_size / batch_size)
    init_data = jnp.zeros([num_batches, batch_size], dtype = jnp.float32)

    def iterate(itr):
      return jax.lax.while_loop(while_pred, while_body, (init_data, itr, 0, False))[0]

    shd_outs = jax.pmap(iterate)(shd_init_s)
    for i in range(jax.local_device_count()):
      seen_data = shd_outs[i]
      d = seen_data.reshape([-1])
      seen_set = set()
      data_set = set([float(x) for x in data])
      for x in d:
        if not jnp.isnan(x):
          assert float(x) not in seen_set
          seen_set.add(float(x))

      assert data_set == seen_set


@pytest.mark.parametrize("batch_size,dataset_size", ALL_SIZES)
def test_lb_correct_pmap(monkeypatch, batch_size, dataset_size):
  with monkeypatch.context() as m:
    m.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4", prepend=',')
    data = jnp.arange(dataset_size, dtype=jnp.float32)
    ds = InMemDataset(data, batch_size, shuffle=True)
    key = jax.random.PRNGKey(0)
    keys = [jax.random.fold_in(key, i) for i in range(jax.local_device_count())]
    list_init_s = [ds.init_state(k) for k in keys]
    shd_init_s = jax.device_put_sharded(list_init_s, jax.local_devices())

    def scan_body(s, _):
      _, _, lb, new_s = ds.next(s)
      return new_s, lb

    num_batches = math.ceil(dataset_size / batch_size)

    def iterate(itr):
      _, lbs = jax.lax.scan(scan_body, itr, jnp.arange(10 * num_batches))
      return lbs

    lbs = jax.pmap(iterate)(shd_init_s)
    true_lbs = jnp.arange(1, (10 * num_batches) + 1) % num_batches == 0
    for i in range(jax.local_device_count()):
      assert jnp.all(lbs[i] == true_lbs)
