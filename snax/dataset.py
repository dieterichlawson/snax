from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import equinox as eqx
from chex import Scalar, Array, ArrayTree
from typing import TypeVar, Generic, Tuple, Callable, Optional
from jax._src.random import KeyArray as PRNGKey

D = TypeVar('D', bound=ArrayTree)
E = TypeVar('E', bound=ArrayTree)

class InMemDataset(eqx.Module, Generic[D]):
  """An in-memory dataset iterator meant for use with JAX."""

  key: PRNGKey
  cursor: Scalar
  inds: Array

  batch_size: int = eqx.static_field()
  shuffle: bool = eqx.static_field()
  num_data_points: int = eqx.static_field()
  num_batches: int = eqx.static_field()
  data: D = eqx.static_field()

  def __init__(self,
          data: D,
          batch_size: int,
          shuffle: bool = False,
          key: Optional[PRNGKey] = None,
          cursor: Optional[Scalar] = None,
          inds: Optional[Array] = None):
    leading_dims = jax.tree_util.tree_map(lambda x: x.shape[0], data)
    leading_dims, _ = jax.tree_util.tree_flatten(leading_dims)
    assert all([x == leading_dims[0] for x in leading_dims]), \
      "Leading dims of all leaves in data are not the same."
    self.num_data_points = leading_dims[0]
    self.batch_size = batch_size
    self.num_batches = math.ceil(float(self.num_data_points) / float(batch_size))
    self.data = data
    self.shuffle = shuffle
    assert not (shuffle and key is None), \
      "To shuffle the dataset, you must provide a key."
    if key is not None:
      self.key = key
    else:
      self.key = jax.random.PRNGKey(0)
    if cursor is not None:
      self.cursor = cursor
    else:
      self.cursor = 0
    if inds is not None:
      self.inds = inds
    else:
      self.inds = self._sample_inds(self.key)


  def _sample_inds(self, key: PRNGKey) -> Array:
    inds_size = self.batch_size * self.num_batches
    if self.shuffle:
      raw_inds = jax.random.choice(
            key,
            self.num_data_points,
            replace=False,
            shape=[self.num_data_points])
      return jnp.pad(raw_inds, (0, inds_size - self.num_data_points))
    else:
      return jnp.arange(inds_size, dtype=jnp.int32)

  def _next_state(self) -> Tuple[Scalar, PRNGKey, Array]:

    def new_state(
            _cursor: Scalar,
            key: PRNGKey,
            _inds: Array) -> Tuple[Scalar, PRNGKey, Array]:
      key, _ = jax.random.split(key)
      return 0, key, self._sample_inds(key)

    new_cursor, new_key, new_inds = jax.lax.cond(
        jnp.greater_equal(self.cursor, self.num_batches - 1),
        new_state,
        lambda c, k, i: (c + 1, k, i),
        self.cursor, self.key, self.inds)
    return new_cursor, new_key, new_inds

  def next(self) -> Tuple[D, Array, Scalar, InMemDataset[D]]:
    start_i = self.cursor * self.batch_size
    data_is = jax.lax.dynamic_slice_in_dim(
        self.inds, start_i, self.batch_size, axis=0)
    indexed_data = jax.tree_util.tree_map(lambda x: x[data_is], self.data)
    last_batch = jnp.equal(self.cursor, self.num_batches - 1)
    batch_remainder = self.num_data_points % self.batch_size
    mask = jnp.where(
        jnp.logical_and(last_batch, batch_remainder > 0),
        jnp.arange(self.batch_size, dtype=jnp.int32) < batch_remainder,
        jnp.ones(self.batch_size, dtype=jnp.int32))
    new_cursor, new_key, new_inds = self._next_state()
    new_ds = InMemDataset(
        self.data, self.batch_size,
        shuffle=self.shuffle, key=new_key, cursor=new_cursor, inds=new_inds)
    return indexed_data, mask, last_batch, new_ds

  def batch_sum_reduce(self, f: Callable[[D], E], init_acc: Optional[E] = None) -> E:

    def mask(m: Array, x: Array) -> Array:
      return jax.numpy.expand_dims(m, list(range(1, x.ndim))) * x

    def while_pred(state: Tuple[Scalar, E, InMemDataset[D]]) -> Scalar:
      last_batch, _, _ = state
      return jnp.logical_not(last_batch)

    def while_body(
            state: Tuple[Scalar, E, InMemDataset[D]]
            ) -> Tuple[Scalar, E, InMemDataset[D]]:
      _, acc, ds = state
      batch, m, lb, ds = ds.next()
      outs = jax.vmap(f)(batch)
      outs_masked = jax.tree_map(lambda x: mask(m, x), outs)
      outs_summed = jax.tree_map(lambda x: jnp.sum(x, axis=0), outs_masked)
      new_acc = jax.tree_map(lambda a, x: a + x, acc, outs_summed)
      return lb, new_acc, ds

    if init_acc is None:
      b, _, _, _ = self.next()
      outs = jax.tree_map(lambda x: jnp.sum(x, axis=0), jax.vmap(f)(b))
      init_acc = jax.tree_map(lambda x: jnp.zeros_like(x), outs)

    _, out, _ = jax.lax.while_loop(
        while_pred, while_body, (jnp.array(False), init_acc, self))

    return out
