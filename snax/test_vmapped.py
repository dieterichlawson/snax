import jax
from . import nn
from . import made
import jax.numpy as jnp

def test_vmapped_affine():
  key = jax.random.PRNGKey(0)
  num_models = 3
  in_dim = 4
  out_dim = 5
  model = nn.VmapModel(key, num_models, nn.Affine, in_dim, out_dim)
  inputs = jnp.ones((num_models, in_dim))
  outs = model(inputs)
  assert outs.shape == (num_models, out_dim)

def test_vmapped_linear():
  key = jax.random.PRNGKey(0)
  num_models = 3
  in_dim = 4
  out_dim = 5
  model = nn.VmapModel(key, num_models, nn.Linear, in_dim, out_dim)
  inputs = jnp.ones((num_models, in_dim))
  outs = model(inputs)
  assert outs.shape == (num_models, out_dim)

def test_vmapped_dense():
  key = jax.random.PRNGKey(0)
  num_models = 3
  in_dim = 4
  out_dim = 5
  model = nn.VmapModel(key, num_models, nn.Dense, in_dim, out_dim, act_fn=jax.nn.relu)
  inputs = jnp.ones((num_models, in_dim))
  outs = model(inputs)
  assert outs.shape == (num_models, out_dim)

def test_vmapped_mlp():
  key = jax.random.PRNGKey(0)
  num_models = 3
  in_dim = 4
  out_dim = 5
  model = nn.VmapModel(key, num_models, nn.MLP, in_dim, layer_dims=[32, 32, out_dim],
          act_fn=jax.nn.relu)
  inputs = jnp.ones((num_models, in_dim))
  outs = model(inputs)
  assert outs.shape == (num_models, out_dim)

def test_vmapped_made():
  key = jax.random.PRNGKey(0)
  num_models = 3
  data_dim = 4
  hidden_dim = 5
  num_res_blocks = 2
  act_fn = jax.nn.relu
  model = nn.VmapModel(key, num_models, made.ResMADE,
          data_dim, hidden_dim, num_res_blocks, act_fn=act_fn)
  inputs = jnp.ones((num_models, data_dim))
  outs = model(inputs)
  assert outs.shape == (num_models, data_dim)
