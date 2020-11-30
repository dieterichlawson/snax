# SNAX

You're hungry for a dead-simple functional deep learning library?

You came to the right place.


### Creating a Multi-layer perceptron example

```
import snax

hidden_sizes = [10, 20, 30]
input_size = 3

mlp = snax.MLP(hidden_sizes)
out_dim, params = mlp.init(jax.random.PRNGKey(0), input_size)
out = mlp.apply(params, jnp.zeros([input_size]))
```

### Creating a deep LSTM example

```
import snax

input_dim = 3
num_steps = 40
hidden_layer_sizes = [32, 64, 32]

lstm = snax.DeepRNN(snax.LSTM, hidden_layer_sizes)

key = jax.random.PRNGKey(0)
_, params = lstm.init(key, input_dim)

# Run the LSTM on some inputs
inputs = jnp.zeros((num_steps, input_dim))
state, outs = dynamic_unroll(params, lstm.apply, inputs, lstm.initial_state())
```
