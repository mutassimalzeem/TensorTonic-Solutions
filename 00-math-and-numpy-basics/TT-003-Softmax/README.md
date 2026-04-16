# TT-003 — Softmax

## Problem Link
TensorTonic: Softmax

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the **softmax function** using NumPy. The function should compute normalized exponential scores over the input and preserve the input shape.

Softmax converts raw scores into probabilities by exponentiating the values and dividing by the sum of exponentials along the last axis.

---

## Official Problem Statement
Implement a vectorized softmax function using NumPy that accepts scalars, Python lists, and NumPy arrays. The result should be a NumPy array of the same shape with values normalized along the last axis.

---

## Constraints
- Input may be a scalar, Python list, or NumPy array
- Output must preserve the input shape
- Use NumPy vectorized operations rather than Python loops
- Apply numerical stability by subtracting the maximum value along the last axis before exponentiation
- The implementation in `solution.py` uses `axis=-1` and `keepdims=True` to support multi-dimensional input

---

## Example

### Input
```python
x = np.array([1.0, 2.0, 3.0])
```

### Output
```python
array([0.09003057, 0.24472847, 0.66524096])
```

---

## Usage

```python
import numpy as np
from solution import softmax

result = softmax([1.0, 2.0, 3.0])
print(result)
```
