# TT-006 - Adam

## Problem Link
TensorTonic: Adam

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement a single **Adam optimizer step** using NumPy.

Adam combines two important optimization ideas:

- Momentum: keeps an exponentially weighted average of gradients.
- RMSProp-style scaling: keeps an exponentially weighted average of squared gradients.

The optimizer then uses bias-corrected versions of these moving averages to update the parameter values.

---

## Official Problem Statement

Write a function that performs one Adam update step for a parameter vector. The function should accept the current parameter values, gradients, first moment estimate `m`, second moment estimate `v`, timestep `t`, and Adam hyperparameters. It should return the updated parameters along with the updated `m` and `v`.

---

## Formula

First moment estimate:

```python
m = beta1 * m + (1 - beta1) * grad
```

Second moment estimate:

```python
v = beta2 * v + (1 - beta2) * grad ** 2
```

Bias correction:

```python
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
```

Parameter update:

```python
param = param - lr * m_hat / sqrt(v_hat + eps)
```

---

## Constraints

- `param`, `grad`, `m`, and `v` may be Python lists or NumPy arrays.
- `param`, `grad`, `m`, and `v` should have compatible shapes.
- `t` should be a positive integer timestep.
- `lr`, `beta1`, `beta2`, and `eps` are scalar hyperparameters.
- The function should return `(param, m, v)`.

---

## Example

### Input

```python
param = [1.0, 2.0]
grad = [0.1, -0.2]
m = [0, 0]
v = [0, 0]
t = 1
lr = 0.001
```

### Output

```python
(array([0.999, 2.001]), array([ 0.01, -0.02]), array([1.e-05, 4.e-05]))
```

---

## Solution Overview

The `solution.py` implementation:

- imports NumPy
- converts `param`, `grad`, `m`, and `v` into float NumPy arrays
- updates the first moment estimate `m`
- updates the second moment estimate `v`
- applies bias correction using timestep `t`
- updates `param` using the Adam update rule
- returns the tuple `(param, m, v)`

The file also includes a sample `print(...)` call at the bottom to demonstrate the function output.

---

## Usage

```python
from solution import adam_step

param, m, v = adam_step(
    param=[1.0, 2.0],
    grad=[0.1, -0.2],
    m=[0, 0],
    v=[0, 0],
    t=1,
    lr=0.001,
)

print(param)
print(m)
print(v)
```
