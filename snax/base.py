from jax._src.random import KeyArray as PRNGKey
from typing import Tuple, Protocol, TypeVar, Optional, Sequence
from chex import Array, ArrayTree

class ParamBase:

  def __iter__(self):
    return self.__dict__.__iter__()

ParamType = TypeVar('ParamType', bound=ArrayTree)

class Layer(Protocol[ParamType]):

  @staticmethod
  def init(key: PRNGKey, input_dim: int) -> Tuple[int, ParamType]:
    """Initialize the layer."""
    ...

  @staticmethod
  def apply(params: ParamType, inputs: Array) -> Array:
    """Apply the layer."""
    ...

StateType = TypeVar('StateType', bound=ArrayTree)

class RecurrentCell(Protocol[ParamType, StateType]):

  @staticmethod
  def init(key: PRNGKey, input_dim: int) -> Tuple[int, ParamType]:
    """Initialize the layer."""
    ...

  @staticmethod
  def apply(params: ParamType, inputs: Array, prev_state: StateType) -> Tuple[StateType, Array]:
    """Apply the layer."""
    ...

  @staticmethod
  def initial_state() -> StateType:
    """Initialize the state."""
    ...

class RecurrentModel(Protocol[ParamType, StateType]):

  @staticmethod
  def init(key: PRNGKey, input_dim: int) -> Tuple[int, ParamType]:
    """Initialize the layer."""
    ...

  @staticmethod
  def apply(params: ParamType,
            inputs: Array,
            length: int,
            initial_state: Optional[StateType]) -> Tuple[StateType, Array]:
    """Apply the layer."""
    ...

  @staticmethod
  def initial_state() -> StateType:
    """Initialize the state."""
    ...

class DeepRecurrentModel(
        RecurrentModel[Sequence[ParamType], Sequence[StateType]],
        Protocol[ParamType, StateType]):
  pass
