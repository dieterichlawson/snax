from __future__ import annotations
import math
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import equinox as eqx
from chex import Scalar, Array, ArrayTree
from typing import TypeVar, Generic, Tuple, Callable, Optional
from jax._src.random import KeyArray as PRNGKey

D = TypeVar('D', bound=ArrayTree)
E = TypeVar('E', bound=ArrayTree)

class IteratorState(eqx.Module):

  key: PRNGKey
  cursor: Scalar
  inds: Array

@dataclass
class InMemDataset(Generic[D]):
  """An in-memory dataset iterator meant for use with JAX under JIT."""

  batch_size: int
  shuffle: bool
  num_data_points: int
  num_batches: int
  data: D

  def __init__(self,
          data: D,
          batch_size: int,
          shuffle: bool = False):
    leading_dims = jax.tree_util.tree_map(lambda x: x.shape[0], data)
    leading_dims, _ = jax.tree_util.tree_flatten(leading_dims)
    assert all([x == leading_dims[0] for x in leading_dims]), \
      "Leading dims of all leaves in data are not the same."
    self.num_data_points = leading_dims[0]
    self.batch_size = batch_size
    self.num_batches = math.ceil(float(self.num_data_points) / float(batch_size))
    self.data = data
    self.shuffle = shuffle

  def init_state(self, key: PRNGKey) -> IteratorState:
    return IteratorState(key, jnp.array(0), self._sample_inds(key))

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

  def _next_state(self, cur_state: IteratorState) -> IteratorState:

    def new_epoch_state(state: IteratorState) -> IteratorState:
      _, key = jax.random.split(state.key)
      return IteratorState(key, jnp.array(0), self._sample_inds(key))

    def same_epoch_state(state: IteratorState) -> IteratorState:
      return IteratorState(state.key, state.cursor + 1, state.inds)

    next_state = jax.lax.cond(
        jnp.greater_equal(cur_state.cursor, self.num_batches - 1),
        new_epoch_state,
        same_epoch_state,
        cur_state)

    return next_state

  def next(self, state: IteratorState) -> Tuple[D, Array, Scalar, IteratorState]:
    start_i = state.cursor * self.batch_size
    data_is = jax.lax.dynamic_slice_in_dim(
        state.inds, start_i, self.batch_size, axis=0)
    indexed_data = jax.tree_util.tree_map(lambda x: x[data_is], self.data)
    last_batch = jnp.equal(state.cursor, self.num_batches - 1)
    batch_remainder = self.num_data_points % self.batch_size
    mask = jnp.where(
        jnp.logical_and(last_batch, batch_remainder > 0),
        jnp.arange(self.batch_size, dtype=jnp.int32) < batch_remainder,
        jnp.ones(self.batch_size, dtype=jnp.int32))
    new_state = self._next_state(state)
    return indexed_data, mask, last_batch, new_state

  def batch_sum_reduce(self,
          f: Callable[[D], E],
          init_acc: Optional[E] = None) -> E:

    def mask(m: Array, x: Array) -> Array:
      return jax.numpy.expand_dims(m, list(range(1, x.ndim))) * x

    def while_pred(state: Tuple[Scalar, E, IteratorState]) -> Scalar:
      last_batch, _, _ = state
      return jnp.logical_not(last_batch)

    def while_body(state: Tuple[Scalar, E, IteratorState]) -> Tuple[Scalar, E, IteratorState]:
      _, acc, s = state
      batch, m, lb, new_s = self.next(s)
      outs = jax.vmap(f)(batch)
      outs_masked = jax.tree_util.tree_map(lambda x: mask(m, x), outs)
      outs_summed = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), outs_masked)
      new_acc = jax.tree_util.tree_map(lambda a, x: a + x, acc, outs_summed)
      return lb, new_acc, new_s

    # We're reducing over the whole dataset so order (and key) doesn't matter.
    init_state = self.init_state(jax.random.PRNGKey(0))

    if init_acc is None:
      b, _, _, _ = self.next(init_state)
      outs = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), jax.vmap(f)(b))
      init_acc = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), outs)

    _, out, _ = jax.lax.while_loop(
        while_pred, while_body, (jnp.array(False), init_acc, init_state))

    return out
