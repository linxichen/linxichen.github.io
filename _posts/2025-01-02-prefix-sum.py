# %%
import jax
import jax.numpy as jnp

@jax.jit
def associative_operator(c_minus, c_t):
    return (c_minus[0] * c_t[0], (c_minus[1] * c_t[0]) + c_t[1])

# %%
T = 200 # time dimension
N = 100 # number of variables
M = 10000 # number of batches/simulations

# %%
seed = 1997
key = jax.random.PRNGKey(seed)
# generate random normal numbers

e_t = jax.random.normal(key, (T, M, N))
# %%
# construct the tuples
As = 0.95*jnp.ones((T, M, N))
# %%
out = jax.lax.associative_scan(associative_operator, (As, e_t), reverse=False, axis=0)
# %%
xs = out[1]
Aout = out[0]

# %%
# compare with sequential simulation code using jax.lax.scan
def ar1_scan(carry, x_t):
    x_prev = carry
    x_new = 0.95 * x_prev + x_t
    return x_new, x_new

initial_x = jnp.zeros((M, N))
_, xs_scan = jax.lax.scan(ar1_scan, initial_x, e_t)

# compute difference
err = jnp.max(jnp.abs(xs - xs_scan))
print(f"The maximum discrepancy is {err}")