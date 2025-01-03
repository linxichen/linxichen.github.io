---
title: "Parallel Simulation of Linear Recurrence Relations via Prefix Sum"
date: 2025-01-02T01:20:00-04:00
categories:
  - Blog
tags:
  - Parallel Simulation
  - GPU
  - Jax
  - Prefix Sum
---

This post explores a clever algorithm by Guy Blelloch for simulating linear recurrence relations in parallel.

## Problem Formulation
Consider the simulation of an AR(1) process:
$$
X_t = A X_{t-1} + \epsilon_t
$$  
The challenge is to simulate this process in parallel using GPUs. Since the process inherently depends on previous values, we need a prefix sum computation. The key is finding a smart operator that is associative, allowing us to compute the prefix sum in parallel over the time dimension.

## The Associative Operator
Let's define an operator $\cdot$ that works on two-tuples of the form $C_t=\left[C_t^a, C_t^b\right]$, where each tuple contains two elements a and b.

The operator $\cdot$ is defined as:
$$
C_{t-1} \cdot C_t=\left[C_{t-1}^a \times C_{t, 1}^a,\left( C_{t-1}^b \times C_t^a \right) +C_t^b \right]
$$  

This operator $\cdot$ produces another two-tuple and is associative (proof in appendix).

## Simulation with the Associative Operator
Without loss of generality, assume $X_0=0$. Define 
$$
C_t = (A, \epsilon_t)
$$  

Let $S_t$ denote the prefix sum at time $t$, with initial value $S_0 = (A^0, X_0)$.

The following steps demonstrate how the operator $\cdot$ computes the prefix sum:

$$
\begin{aligned}
S_1 & =S_0 \cdot C_1 \\
& =\left[1, x_0\right] \cdot\left[A, \epsilon_1\right] \\
& =\left[A, x_0 A+\epsilon_1\right] \\
& =\left[A, x_1\right] . \\
S_2 & =S_1 \cdot C_2 \\
& =\left[A, x_1\right] \cdot\left[A, \epsilon_2\right] \\
& =\left[A^2, A x_1+\epsilon_2\right] \\
S_3 & =S_2 \cdot C_3 \\
& =\left[A^2, x_2\right] \cdot\left[A, \epsilon_3\right] \\
& =\left[A^3, A x_2+\epsilon_3\right]
\end{aligned}
$$  

We can observe that $S_t$ consists of two sequences: the first sequence ($y_t$) represents the cumulative products of $A$, while the second sequence is our desired $X_t$:
$$
\begin{aligned}
S_t & =\left[y_t, x_t\right] \\
&\overset{\text{def}}{=} {\left[y_{t-1} \times a_t,\left(x_{t-1} \times a_t\right) + \epsilon_t\right] } \\
& =\left[y_{t-1} \times c_t^a,\left(x_{t-1} \times c_t^a\right) + c_t^b\right] \\
& =\left[y_{t-1}, x_{t-1}\right] \cdot c_t \\
& =S_{t-1} \cdot c_t
\end{aligned}
$$  

# Implementation in JAX
The following code demonstrates the implementation using JAX's pytree to represent the two-tuple and `jax.lax.associative_scan` for parallel prefix sum computation. We verify correctness by comparing with sequential simulation using `jax.lax.scan`.

```python
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
```

# Appendix
## Verification of Associative Property

$$
\begin{aligned}
\left(S_1 \cdot c_2\right) \cdot c_3 & =s_2 \cdot c_3=S_3 \\
S_1 \cdot\left(c_2 \cdot c_3\right) & =S_1 \cdot\left(\left[A, \epsilon_2\right] \cdot\left[A, \epsilon_3\right]\right) \\
& =S_1 \cdot\left(\left[A^2, A \epsilon_2+\epsilon_3\right]\right) \\
& =\left[A, x_1\right] \cdot\left[A^2, A \epsilon_2+\epsilon_3\right] \\
& =\left[A^3, A^2 x_1+A \epsilon_2+\epsilon_3\right] \\
& =\left[A^3, x_3^3\right]
\end{aligned}
$$  

This equality holds because:
$$
\begin{aligned}
x_3 & =A x_2+\epsilon_3 \\
& =A\left[A x_1+\epsilon_2\right]+\epsilon_3 \\
& =A^2 x_1+A \epsilon_2+\epsilon_3
\end{aligned}
$$