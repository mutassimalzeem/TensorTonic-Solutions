# TT-002 — Mean Squared Error (NumPy)

## Problem Link
TensorTonic: Mean Squared Error (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the **mean squared error (MSE)** function using NumPy. The function should compute the average of the squared differences between predicted values and true values.

The mean squared error is defined as:

```
MSE = mean((y_pred - y_true)^2)
```

---

## Official Problem Statement
Implement a NumPy-based MSE function that accepts numeric sequences such as Python lists and NumPy arrays. The function should return the mean squared error when the prediction and target arrays have the same shape.

---

## Constraints
- Input arrays must have the same shape.
- Use NumPy operations rather than Python loops.
- The function should accept scalars, Python lists, and NumPy arrays.
- If the input shapes do not match, the implementation returns `None`.

---

## Example

### Input
```python
y_pred = np.array([2.0, 3.0, 4.0])
y_true = np.array([1.0, 2.0, 3.0])
```

### Output
```python
1.0
```

---

## Usage

```python
import numpy as np
from solution import mean_squared_error

result = mean_squared_error([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
print(result)
```
