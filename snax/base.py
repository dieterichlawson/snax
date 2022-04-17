from typing import Tuple, TypeVar

try:
  from typing import Protocol
except ImportError:
  from typing_extensions import Protocol

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
