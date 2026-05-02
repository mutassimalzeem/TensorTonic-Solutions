# TT-021 - Logistic Regression

## Problem Link
TensorTonic: Logistic Regression

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **binary logistic regression** using NumPy and train the model with gradient descent.

For each sample, the model computes:

```python
z = X @ w + b
p = sigmoid(z)
```

The predicted probability `p` is then compared with the binary target labels using binary cross entropy.

---

## Official Problem Statement
Implement a NumPy-based logistic regression trainer that learns weights and bias from input features and binary labels using gradient descent.

---

## Constraints
- `X` should be a 2D feature matrix with shape `(n_samples, n_features)`.
- `y` should be a 1D binary label array with values `0` or `1`.
- Initialize weights with zeros and bias with `0.0`.
- Use the sigmoid function to convert logits into probabilities.
- Use binary cross entropy gradients for the weight and bias updates.
- Return the learned `(w, b)` pair.

---

## Example

### Input
```python
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=500)
```

### Output
```python
(array([2.527...]), -3.430...)
```

The exact values may vary slightly with learning rate and number of steps.

---

## Solution Overview

The implementation in `solution.py`:

- converts `X` and `y` to NumPy float arrays
- initializes `w` as zeros and `b` as `0.0`
- computes logits with `np.dot(X, w) + b`
- applies a numerically stable sigmoid function
- clips probabilities before using logarithms
- computes binary cross entropy for the batch
- computes vectorized gradients for `w` and `b`
- updates parameters using gradient descent
- returns `(w, b)`

---

## Usage

```python
from solution import train_logistic_regression

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

w, b = train_logistic_regression(X, y, lr=0.1, steps=500)
print(w, b)
```
