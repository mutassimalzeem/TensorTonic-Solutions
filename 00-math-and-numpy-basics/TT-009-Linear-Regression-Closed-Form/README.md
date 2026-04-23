# TT-009 — Linear Regression Closed Form (NumPy)

## Problem Link
TensorTonic: Linear Regression Closed Form (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **closed-form linear regression** to find optimal weights w for X w ≈ y.

The closed form solution is:

```
w = (X^T X)^(-1) X^T y
```

---

## Official Problem Statement
Implement linear regression weights using normal equation (closed form), assuming X includes intercept column.

---

## Constraints
- X: n x (d+1) design matrix (with bias column)
- y: n x 1 target vector
- Output: (d+1,) weights vector
- Assumes X^T X invertible (no regularization)
- Uses matrix operations only

---

## Example

### Input
```python
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 3, 5])
```

### Output
```python
array([0. , 1.5])  # w0=0, w1=1.5 approx
```

---

## Solution Overview

The implementation in `solution.py`:

- `w1 = np.linalg.inv(np.dot(np.transpose(X), X))`
- `w2 = np.dot(np.transpose(X), y)`
- `w = np.dot(w1, w2)`

Matrix form of normal equation.

---

## Usage

```python
import numpy as np
from solution import linear_regression_closed_form

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 3, 5])
w = linear_regression_closed_form(X, y)
print(w)
```

