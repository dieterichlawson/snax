from . import made
import jax
import jax.numpy as jnp

def test_resmade():
  data_dim = 3
  hidden_dim = 20
  num_res_blocks = 3
  outputs_per_dim = 1
  key = jax.random.PRNGKey(0)
  md = made.ResMADE(key, data_dim, hidden_dim, num_res_blocks, outputs_per_dim, act_fn=None)

  x = jnp.ones(data_dim)

  def grad_of_out_i_wrt_in_j(x, out_i, in_j):
    return jax.grad(lambda a: md(a)[out_i])(x)[in_j]

  for i in range(data_dim):
    for j in range(data_dim):
      if i <= j:
        assert grad_of_out_i_wrt_in_j(x, i, j) == 0.
      else:
        assert grad_of_out_i_wrt_in_j(x, i, j) != 0.

def test_resmade_multi_d_out():
  data_dim = 3
  hidden_dim = 20
  num_res_blocks = 3
  key = jax.random.PRNGKey(0)
  md = made.ResMADE(key, data_dim, hidden_dim, num_res_blocks, 
          inputs_per_dim=1,
          outputs_per_dim=2,
          act_fn=None)

  x = jnp.ones(data_dim)

  def grad_of_out_i_wrt_in_j(x, out_i, in_j, out_d):
    return jax.grad(lambda a: md(a)[out_i, out_d])(x)[in_j]

  for i in range(data_dim):
    for j in range(data_dim):
      if i <= j:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0) == 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1) == 0.
      else:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0) != 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1) != 0.

def test_resmade_multi_d_in():
  data_dim = 3
  hidden_dim = 20
  num_res_blocks = 3
  inputs_per_dim = 2
  key = jax.random.PRNGKey(0)
  md = made.ResMADE(key, data_dim, hidden_dim, num_res_blocks,
          inputs_per_dim=inputs_per_dim,
          outputs_per_dim=1,
          act_fn=None)

  x = jnp.ones([data_dim, inputs_per_dim])

  def grad_of_out_i_wrt_in_j(x, out_i, in_j, in_d):
    return jax.grad(lambda a: md(a)[out_i])(x)[in_j, in_d]

  for i in range(data_dim):
    for j in range(data_dim):
      if i <= j:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0) == 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1) == 0.
      else:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0) != 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1) != 0.

def test_resmade_multi_d_in_out():
  data_dim = 3
  hidden_dim = 20
  num_res_blocks = 3
  inputs_per_dim = 2
  outputs_per_dim = 2
  key = jax.random.PRNGKey(0)
  md = made.ResMADE(key, data_dim, hidden_dim, num_res_blocks,
          inputs_per_dim=inputs_per_dim,
          outputs_per_dim=outputs_per_dim,
          act_fn=None)

  x = jnp.ones([data_dim, inputs_per_dim])

  def grad_of_out_i_wrt_in_j(x, out_i, in_j, out_d, in_d):
    return jax.grad(lambda a: md(a)[out_i, out_d])(x)[in_j, in_d]

  for i in range(data_dim):
    for j in range(data_dim):
      if i <= j:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0, 0) == 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1, 0) == 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 0, 1) == 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1, 1) == 0.
      else:
        assert grad_of_out_i_wrt_in_j(x, i, j, 0, 0) != 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 0, 1) != 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1, 0) != 0.
        assert grad_of_out_i_wrt_in_j(x, i, j, 1, 1) != 0.


