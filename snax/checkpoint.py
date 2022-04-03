import dill as pickle
from pathlib import Path
from typing import TypeVar, Any, List, Optional, Tuple
import types
import dataclasses
import jax
import jaxlib

Data = TypeVar('Data')

def save_checkpoint(
        data: Any,
        step: int,
        checkpoint_dir: str,
        num_checkpoints_to_keep: int = 3,
        name_prefix: str = "checkpoint",
        step_format: str = "{step:08d}",
        filetype: str = ".chk") -> None:
  existing_checkpoints = get_checkpoints(
          checkpoint_dir,
          name_prefix=name_prefix,
          filetype=filetype)
  checkpoint_steps = [step_from_path(f) for f in existing_checkpoints]
  assert len(checkpoint_steps) <= num_checkpoints_to_keep, "Too many checkpoints saved."
  assert len(checkpoint_steps) == len(set(checkpoint_steps)), "Non-unique checkpoint step."
  assert all([step > s for s in checkpoint_steps]), "Attempting to write earlier checkpoint."
  if len(checkpoint_steps) == num_checkpoints_to_keep:
    to_remove = checkpoint_steps.index(min(checkpoint_steps))
    existing_checkpoints[to_remove].unlink()
  new_name = name_prefix + "_" + step_format.format(step=step) + filetype
  new_path = Path(checkpoint_dir) / new_name
  save_checkpoint_to_path(data, step, new_path)

def get_checkpoints(
        checkpoint_dir: str,
        name_prefix: str = "checkpoint",
        filetype: str = ".chk") -> List[Path]:
  checkpoint_glob = name_prefix + '*' + filetype
  checkpoint_paths = Path(checkpoint_dir).glob(checkpoint_glob)
  return list(checkpoint_paths)

def step_from_path(path: Path) -> int:
  return int(path.stem.split("_")[1])

def get_latest_checkpoint_path(
        checkpoint_dir: str,
        name_prefix: str = "checkpoint",
        filetype: str = ".chk") -> Optional[Path]:
  paths = get_checkpoints(checkpoint_dir, name_prefix=name_prefix, filetype=filetype)
  if len(paths) == 0:
    return None
  steps = [step_from_path(x) for x in paths]
  max_ind = steps.index(max(steps))
  return paths[max_ind]

def load_latest_checkpoint(
        checkpoint_dir: str,
        name_prefix: str = "checkpoint",
        filetype: str = ".chk") -> Optional[Tuple[Any, int]]:
  path = get_latest_checkpoint_path(checkpoint_dir, name_prefix=name_prefix, filetype=filetype)
  if path is None:
    return None
  return load_checkpoint_from_path(path)

def load_checkpoint_from_path(path: Path) -> Tuple[Any, int]:
  with path.open(mode='rb') as f:
    data, step = pickle.load(f)
  return data, step

def save_checkpoint_to_path(data: Any, step: int, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open(mode='wb') as f:
    pickle.dump((data, step), f)


def get_static_fields(model):
  fields = set()
  for f in dataclasses.fields(model):
    if 'static' in f.metadata and f.metadata['static']:
      fields.add(f.name)
  return fields


def assert_static_fields_same(x, y):
  x_fields = get_static_fields(x)
  y_fields = get_static_fields(y)
  x_dict = dataclasses.asdict(x)
  y_dict = dataclasses.asdict(y)
  assert x_fields == y_fields
  for f in x_fields:
    x_f = x_dict[f]
    y_f = y_dict[f]
    if (isinstance(x_f, types.FunctionType) or
        isinstance(x_f, jaxlib.xla_extension.CompiledFunction)):
      # It's a function so there's nothing we can do.
      pass
    else:
      assert x_f == y_f, "%s != %s" % (x_dict[f], y_dict[f])


def load_latest_checkpoint_with_model(
        model: Any,
        checkpoint_dir: str,
        name_prefix: str = "checkpoint",
        filetype: str = ".chk") -> Optional[Tuple[Any, int]]:
  path = get_latest_checkpoint_path(checkpoint_dir, name_prefix=name_prefix, filetype=filetype)
  if path is None:
    return None
  _, model_treedef = jax.tree_util.tree_flatten(model)
  new_model, step = load_checkpoint_from_path(path)
  loaded_leaves, _ = jax.tree_util.tree_flatten(new_model)
  restored_model = model_treedef.unflatten(loaded_leaves)
  assert_static_fields_same(model, new_model)
  return restored_model, step
