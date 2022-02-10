import pickle
from pathlib import Path

def save_checkpoint(data, step, checkpoint_dir,
        num_checkpoints_to_keep=3, name_prefix="checkpoint",
        step_format="{step:08d}", filetype=".chk"):
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

def get_checkpoints(checkpoint_dir, name_prefix="checkpoint", filetype=".chk"):
  checkpoint_glob = name_prefix + '*' + filetype
  checkpoint_paths = Path(checkpoint_dir).glob(checkpoint_glob)
  return list(checkpoint_paths)

def step_from_path(path):
  return int(path.stem.split("_")[1])

def get_latest_checkpoint_path(checkpoint_dir, name_prefix="checkpoint", filetype=".chk"):
  paths = get_checkpoints(checkpoint_dir, name_prefix=name_prefix, filetype=filetype)
  if len(paths) == 0:
    return None
  steps = [step_from_path(x) for x in paths]
  max_ind = steps.index(max(steps))
  return paths[max_ind]

def load_latest_checkpoint(checkpoint_dir, name_prefix="checkpoint", filetype=".chk"):
  path = get_latest_checkpoint_path(checkpoint_dir, name_prefix=name_prefix, filetype=filetype)
  if path is None:
    return None
  return load_checkpoint_from_path(path)

def load_checkpoint_from_path(path):
  with path.open(mode='rb') as f:
    data, step = pickle.load(f)
  return data, step

def save_checkpoint_to_path(data, step, path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open(mode='wb') as f:
    pickle.dump((data, step), f)
