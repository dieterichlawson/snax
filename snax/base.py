from typing import Tuple, TypeVar, Protocol

from chex import Array

StateType = TypeVar('StateType')

class RecurrentCell(Protocol[StateType]):

  out_dim: int

  def __call__(self, prev_state: StateType, inputs: Array) -> Tuple[StateType, Array]:
    """Apply the layer."""
    ...

  def initial_state(self) -> StateType:
    """Initialize the state."""
    ...
