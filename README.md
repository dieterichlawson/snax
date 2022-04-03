# SNAX

Hungry for a dead-simple functional deep learning library?

You came to the right place.


### Creating a Multi-layer perceptron

```
import jax
import jax.numpy as jnp
import snax

hidden_sizes = [10, 20, 30]
input_size = 3
key = jax.random.PRNGKey(0)

mlp = snax.nn.MLP(key,
                  input_size,
                  hidden_sizes,
                  act_fn=jax.nn.relu)
out = mlp(jnp.ones([input_size]))
```

### Creating a deep LSTM

```
import jax
import jax.numpy as jnp
import snax

input_size = 3
num_steps = 40
hidden_layer_sizes = [32, 64, 32]
key = jax.random.PRNGKey(0)

lstm = snax.recurrent.LSTM(key,
                           input_size,
                           hidden_layer_sizes,
                           act_fn=jnp.tanh,
                           forget_gate_bias_init=1.)


# Run the LSTM on some inputs
inputs = jnp.zeros((num_steps, input_size))
new_state, outs = LSTM(inputs, lstm.initial_state())
```
