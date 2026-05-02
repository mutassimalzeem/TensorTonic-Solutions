# TT-004 — Cross Entropy

## Problem Link
TensorTonic: Cross Entropy

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the **cross entropy loss** for one-hot labels using NumPy. The function should compute the average negative log-likelihood of the predicted class probabilities with respect to the true class labels.

Cross entropy for one sample is:

```
CE = - sum(y_true * log(y_pred))
```

And the batch loss is the mean across all samples.

---

## Official Problem Statement
Implement a NumPy-based cross entropy loss function that accepts either one-hot encoded labels or integer class labels and predicted probability distributions. The implementation should be vectorized and should handle numeric sequence inputs such as Python lists and NumPy arrays.

---

## Constraints
- `y_pred` must have shape `(n_samples, n_classes)`.
- `y_true` may be one-hot encoded with the same shape as `y_pred`, or a 1D array of integer class labels.
- `y_pred` should contain class probabilities that sum to 1 along the last axis.
- Use NumPy vectorized operations rather than Python loops.
- Clip predictions before taking the logarithm to avoid `log(0)`.

---

## Example

### Input
```python
y_true = [[1, 0, 0], [0, 1, 0]]
y_pred = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
```

### Output
```python
0.1602501
```

---

## Solution Overview

The `solution.py` implementation:

- converts `y_true` and `y_pred` to NumPy arrays
- clips `y_pred` values to the range `[1e-15, 1 - 1e-15]`
- if `y_true` and `y_pred` have the same shape, computes the one-hot cross entropy using `y_true * np.log(y_pred)`
- if `y_true` is 1D, selects the predicted probability for each true class index
- averages the per-sample losses with `np.mean`

This returns the mean cross entropy loss for the batch.

---

## Usage

```python
import numpy as np
from solution import cross_entropy_loss

labels = [[1, 0, 0], [0, 1, 0]]
predictions = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
print(cross_entropy_loss(labels, predictions))
```
