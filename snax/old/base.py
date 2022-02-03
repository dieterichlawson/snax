from jax._src.random import KeyArray as PRNGKey
from typing import Tuple, Protocol, TypeVar, Optional, Sequence
from chex import Array, ArrayTree

class Module(object):
  """A wrapper for a collection of related functions."""

  def __init__(self, *args):
    for fn in args:
      setattr(self, fn.__name__, fn)
