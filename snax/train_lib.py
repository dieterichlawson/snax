from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from optax import Params, OptState
from jax._src.random import KeyArray as PRNGKey
from chex import Scalar, ArrayTree, Array
from typing import Optional, TypeVar
from timeit import default_timer as timer
import wandb
from typing import Callable, Tuple, List

from . import checkpoint as chk
from .dataset import InMemDataset, IteratorState, DummyDataset

D = TypeVar('D', bound=ArrayTree)
S = TypeVar('S')

LossFn = Callable[[PRNGKey, int, Params], Scalar]
LossFnWithData = Callable[[PRNGKey, int, Params, D], Scalar]
TrainStepFn = Callable[
        [PRNGKey, int, Params, OptState],
        Tuple[Params, OptState, Scalar]]
SummarizeFn = Callable[[PRNGKey, Params, int], None]

def avg_batch_loss(
        loss_fn: LossFn,
        batch_size,
        key: PRNGKey,
        step: int,
        params: Params) -> float:
  keys = jax.random.split(key, num=batch_size)
  losses = jax.vmap(loss_fn, in_axes=(0, None, None))(keys, step, params)
  return jnp.mean(losses)

def avg_batch_loss_with_data(
        loss_fn: LossFnWithData,
        batch_size,
        key: PRNGKey,
        step: int,
        params: Params,
        data,
        mask: Array) -> float:
  keys = jax.random.split(key, num=batch_size)
  losses = jax.vmap(loss_fn, in_axes=(0, None, None, 0))(keys, step, params, data)
  return jnp.mean(mask * losses)

def repeat_step(
      step_fn,
      num_inner_steps: int,
      step: int,
      params: Params,
      state: S) -> Tuple[Params, float, S]:

  def for_step(step: int, carry: Tuple[Params, float, S]) -> Tuple[Params, float, S]:
    params, _, state = carry
    return step_fn(step, params, state)

  return jax.lax.fori_loop(step, step + num_inner_steps, for_step, (params, 0., state))


class TrainStepState(eqx.Module):
  key: PRNGKey
  opt_state: OptState
  itr_state: IteratorState


class TrainStep:
  """A function that minimizes a LossFn."""

  def __init__(
      self,
      loss_fn,
      optimizer: optax.GradientTransformation,
      dataset: InMemDataset = None,
      parallelize: bool = False,
      num_inner_steps: int = 1,
      batch_size: int = 1,
      name: str = "loss"):
    self.num_inner_steps = num_inner_steps
    self.parallelize = parallelize
    self.optimizer = optimizer
    self.name = name
    if dataset is not None:
      assert parallelize or (batch_size == dataset.batch_size), \
              "dataset batch size does not equal supplied batch size."
    if parallelize:
      assert batch_size % jax.local_device_count() == 0, \
        f"num devices ({jax.local_device_count()}) must evenly divide batch_size ({batch_size})."
      local_batch_size = batch_size // jax.local_device_count()
    else:
      local_batch_size = batch_size

    # Possibly adapt the loss to take a data point.
    if dataset is None:
      self.dataset = DummyDataset()
      loss_fn_with_data = lambda k, s, p, _: loss_fn(k, s, p)
    else:
      loss_fn_with_data = loss_fn
      self.dataset = dataset

    vm_loss_fn = jax.vmap(loss_fn_with_data, in_axes=(0, None, None, 0))

    def local_batch_loss(
            key: PRNGKey,
            step: int,
            params: Params,
            data,
            mask: Array) -> float:
      keys = jax.random.split(key, num=local_batch_size)
      losses = vm_loss_fn(keys, step, params, data)
      if mask is not None:
        return jnp.sum(jnp.where(mask, x=losses, y=jnp.zeros_like(losses))) / jnp.sum(mask)
      else:
        return jnp.mean(losses)

    grad_loss = jax.value_and_grad(local_batch_loss, argnums=2)

    def _step_fn(
          step: int,
          params: optax.Params,
          state: TrainStepState) -> Tuple[Params, float, TrainStepState]:
      key, subkey = jax.random.split(state.key)
      data, mask, _, new_itr_state =  self.dataset.next(state.itr_state)
      loss_val, grads = grad_loss(key, step, params, data, mask)
      if parallelize:
        loss_val = jax.lax.pmean(loss_val, 'b')
        grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, 'b'), grads)
      updates, new_opt_state = optimizer.update(grads, state.opt_state, params=params)
      params = optax.apply_updates(params, updates)
      return params, loss_val, TrainStepState(subkey, new_opt_state, new_itr_state)

    if num_inner_steps > 1:
      step_fn = partial(repeat_step, _step_fn, num_inner_steps)
    else:
      step_fn = _step_fn

    if parallelize:
      self.step_fn = jax.pmap(step_fn, axis_name='b', in_axes=(None, None, 0))
    else:
      self.step_fn = jax.jit(step_fn)


  def init_state(self, key, init_params) -> TrainStepState:
    num_devices = jax.local_device_count()
    state_list = []
    for i in range(num_devices):
      k = jax.random.fold_in(key, i)
      k, sk = jax.random.split(k)
      itr_state = self.dataset.init_state(sk)
      opt_state = self.optimizer.init(init_params)
      state_list.append(TrainStepState(k, opt_state, itr_state))

    if self.parallelize:
      res = jax.device_put_sharded(state_list, jax.local_devices())
      return res
    else:
      return state_list[0]

  def __call__(
          self,
          step: int,
          params: Params,
          state: TrainStepState) -> Tuple[Params, Scalar, TrainStepState]:
    """Run a step of training.

    Args:
      key: A JAX PRNGKey. If parallelize is true, should either be sharded or have leading
        dimension of jax.local_device_count().
      step: The current training step, the number of times this TrainStep has been called.
        Does not include inner steps or steps from other losses.
      params: The parameters to be optimized.
      opt_state: The optimizer state. If parallelize is true, should either be sharded or have
        leading dimension of jax.local_device_count().

    Returns:
      key: A new JAX PRNGKey. If parallelize is True, will be sharded among jax.local_devices().
      params: Updated parameters. Will not be sharded.
      opt_state: The new optimizer state. If parallelize is true, will be sharded.
      loss_val: The loss value of the most recent inner step. Will not be sharded.
    """
    new_params, loss_val, new_state = self.step_fn(step, params, state)
    if self.parallelize:
      new_params, loss_val = jax.tree_util.tree_map(lambda x: x[0], (new_params, loss_val))
    return new_params, loss_val, new_state


def train_alternating(
        key: PRNGKey,
        train_steps: List[TrainStep],
        init_params: Params,
        num_steps: int = 100,
        summarize_fn: Optional[SummarizeFn] = None,
        summarize_every: int = 100,
        checkpoint_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoints_to_keep: int = 3,
        use_wandb: bool = False) -> Params:
  """Run training.

  Iteratively runs a set of training steps, logging performance metrics and computing summaries.

  Args:
    key: A JAX PRNGKey.
    train_steps: A list of TrainSteps.
    init_params: The initial parameters.
    num_steps: The number of steps to run training for.
    summarize_fn: A function that computes and logs summaries. Must accept a key,
      the current parameters, and the current step.
    summarize_every: The number of steps between calls to summarize_fn.
    checkpoint_every: The number of steps between checkpoints.
    checkpoint_dir: The directory to store checkpoints.
    checkpoints_to_keep: The number of recent checkpoints to keep in the checkpoint directory.
    use_wanbd: Whether to use weights and biases to log performance metrics like steps per
      second and the number of seconds the summary_fn takes.
  Returns:
    The parameters after training for num_steps.
  """
  # Check that summarize_every and checkpoint_every are multiples of the number of inner steps.
  total_inner_steps = sum([ts.num_inner_steps for ts in train_steps])
  if summarize_fn is not None:
    assert summarize_every % total_inner_steps == 0, \
            f"summarize_every ({summarize_every}) must be a multiple of the" \
            f" total number of inner steps ({total_inner_steps})"
  if checkpoint_every is not None:
    assert checkpoint_every % total_inner_steps == 0, \
            f"checkpoint_every ({checkpoint_every}) must be a multiple of the" \
            f" total number of inner steps ({total_inner_steps})"

  assert len(train_steps) == len(set([ts.name for ts in train_steps])), \
          "All train steps must have a unique name."

  # Set a dummy summarize_fn if it was not provided
  if summarize_fn is None:
    summarize_fn = lambda *_: None

  # Initialize the parameters and step states
  params = init_params
  global_start_step = 0
  local_steps = [0] * len(train_steps)
  keys = jax.random.split(key, num=len(train_steps))
  step_states = [ts.init_state(keys[i], init_params) for i, ts in enumerate(train_steps)]

  # Maybe load a checkpoint.
  if checkpoint_dir is not None and checkpoint_every is not None:
    out = chk.load_latest_checkpoint((params, step_states, local_steps), checkpoint_dir)
    if out[0] is not None:
      (params, step_states, local_steps), global_start_step = out
      print(f"Loaded checkpoint at step {global_start_step} from {checkpoint_dir}.")
    else:
      print("Checkpoint not found.")

  # Summarize on the first step.
  print(f"Step {global_start_step}")
  key, subkey = jax.random.split(key)
  summarize_fn(subkey, params, global_start_step)

  # Train.
  step = global_start_step
  while step < num_steps:
    new_step_states = []
    new_local_steps = []
    metrics = {}
    # Run the train_fns for one outer step each.
    for train_step_fn, state, local_step in zip(train_steps, step_states, local_steps):
      start_time = timer()
      # Run one of the train steps
      params, loss_val, new_state = train_step_fn(local_step, params, state)
      loss_val.block_until_ready()
      sec = timer() - start_time
      step_metrics = {'perf/steps_per_sec': train_step_fn.num_inner_steps / sec,
                      'loss': loss_val}
      metrics[train_step_fn.name] = step_metrics
      step += train_step_fn.num_inner_steps
      new_local_steps.append(local_step + train_step_fn.num_inner_steps)
      new_step_states.append(new_state)
    step_states = new_step_states
    local_steps = new_local_steps

    # Possibly summarize and save a checkpoint.
    if step != global_start_step:
      if step % summarize_every == 0:
        # Print losses.
        print(f"Step {step}")
        for loss_name, loss_metrics in metrics.items():
          loss_val = loss_metrics['loss']
          sps = loss_metrics['perf/steps_per_sec']
          print(f"  {loss_name}: {loss_val:0.3f}, steps/sec {sps:0.2f}")
        # Compute summaries
        summ_start_time = timer()
        key, subkey = jax.random.split(key)
        summarize_fn(subkey, params, step)
        summ_elapsed_time = timer() - summ_start_time
        metrics['summ_secs'] = summ_elapsed_time
        print(f"  summary sec: {summ_elapsed_time:0.2f}")
        # Log to wandb
        if use_wandb: wandb.log(metrics, step=step)

      if (checkpoint_dir is not None
          and checkpoint_every is not None
          and step % checkpoint_every == 0):
        print(f"Saving checkpoint for step {step} at {checkpoint_dir}... ", end="")
        chk.save_checkpoint((params, step_states, local_steps), step, checkpoint_dir,
                            num_checkpoints_to_keep=checkpoints_to_keep)
        print("Done.")

  return params

def train(key: PRNGKey,
          loss_fn: LossFn,
          optimizer: optax.GradientTransformation,
          init_params: Params,
          dataset: InMemDataset = None,
          num_steps: int = 100,
          num_inner_steps: int = 1,
          parallelize: bool = False,
          batch_size: int = 1,
          loss_name: str = "loss",
          summarize_fn: Optional[SummarizeFn] = None,
          summarize_every: int = 100,
          checkpoint_every: Optional[int] = None,
          checkpoint_dir: Optional[str] = None,
          checkpoints_to_keep: int = 3,
          use_wandb: bool = False) -> Params:
  """Run training.

  Iteratively runs a training step, logging performance metrics and computing summaries.

  Args:
    key: A JAX PRNGKey.
    loss_fn: A callable LossFn that computes a scalar loss given a PRNGKey, the current step,
      and a set of parameters.
    optimizer: An optax optimizer.
    init_params: The initial parameters.
    num_steps: The number of steps to run training for.
    parallelize: If true, parallelize batched gradient evaluation over
      the JAX local devices using pmap.
    batch_size: The size of the batch, the number of parallel evaluations of loss_fn to
      perform.
    summarize_fn: A function that computes and logs summaries. Must accept a key,
      the current parameters, and the current step.
    summarize_every: The number of steps between calls to summarize_fn.
    checkpoint_every: The number of steps between checkpoints.
    checkpoint_dir: The directory to store checkpoints.
    checkpoints_to_keep: The number of recent checkpoints to keep in the checkpoint directory.
    use_wanbd: Whether to use weights and biases to log performance metrics like steps per
      second and the number of seconds the summary_fn takes.
  Returns:
    The parameters after training for num_steps.
  """
  train_step = TrainStep(
          loss_fn, optimizer,
          dataset=dataset,
          num_inner_steps=num_inner_steps,
          name=loss_name,
          parallelize=parallelize,
          batch_size=batch_size)
  return train_alternating(key,
                           [train_step],
                           init_params,
                           num_steps=num_steps,
                           summarize_fn=summarize_fn,
                           summarize_every=summarize_every,
                           checkpoint_every=checkpoint_every,
                           checkpoint_dir=checkpoint_dir,
                           checkpoints_to_keep=checkpoints_to_keep,
                           use_wandb=use_wandb)
