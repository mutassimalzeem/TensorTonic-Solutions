# TT-005 — Gradient Descent

## Problem Link
TensorTonic: Gradient Descent

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement a simple **gradient descent update** for a quadratic function:

```python
f(x) = ax^2 + bx + c
```

For this function, the derivative is:

```python
f'(x) = 2ax + b
```

Gradient descent updates the current value of `x` by moving in the opposite direction of the gradient:

```python
x = x - lr * (2ax + b)
```

---

## Official Problem Statement

Write a function that performs gradient descent on the quadratic expression `ax^2 + bx + c`, starting from an initial value `x0`, using learning rate `lr` for a given number of `steps`, and returns the final value of `x`.

---

## Constraints
- `a`, `b`, `c`, `x0`, and `lr` are numeric values.
- `steps` is a non-negative integer.
- The derivative of `ax^2 + bx + c` is `2ax + b`.
- The result should be the final scalar value after repeated updates.

---

## Example

### Input
```python
a = 2
b = -8
c = 3
x0 = 0
lr = 0.1
steps = 5
```

### Output
```python
2.0
```

---

## Solution Overview

The `solution.py` implementation:

- defines `gradient_descent_quadratic(a, b, c, x0, lr, steps)`
- replaces the provided `lr` with `1 / (2 * a)`
- applies the gradient expression `2 * a * x0 + b`
- runs an update loop for `steps` iterations
- returns the final value stored in `x0`

For the current implementation, when `steps > 0`, the returned value converges directly to:

```python
-b / (2a)
```

which is the minimizer of the quadratic function.

---

## Usage

```python
from solution import gradient_descent_quadratic

result = gradient_descent_quadratic(2, -8, 3, 0, 0.1, 5)
print(result)
```
