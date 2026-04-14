# TT-001 — Sigmoid (NumPy)

## Problem Link
TensorTonic: Sigmoid (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the **sigmoid function** using NumPy so that it works for scalar and array inputs.
The function should compute the sigmoid activation for every element and preserve the input shape.

The sigmoid function is defined as:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

---

## Official Problem Statement
Implement a vectorized sigmoid function using NumPy that accepts scalars, Python lists, and NumPy arrays.

---

## Constraints
- Input may be a scalar, Python list, or NumPy array
- Output must preserve the input shape
- Use NumPy vectorized operations rather than Python loops
- The implementation in `solution.py` converts input with `np.array(x)` and computes sigmoid directly

---

## Example

### Input
```python
x = np.array([-1.0, 0.0, 1.0])
```

### Output
```python
array([0.26894142, 0.5, 0.73105858])
```

---

## Solution Overview

The implementation in `solution.py`:

- converts the input to a NumPy array with `np.array(x)`
- computes `1 / (1 + np.exp(-x))`
- returns the result as a NumPy array with the same shape

This approach is vectorized and works for both scalar and array inputs.

---

## Usage

```python
import numpy as np
from solution import sigmoid

result = sigmoid(np.array([-1.0, 0.0, 1.0]))
print(result)
```