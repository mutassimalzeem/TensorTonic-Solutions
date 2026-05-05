# TT-024 - PCA Projection

## Problem Link
TensorTonic: PCA Projection

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **principal component analysis projection** using NumPy. Given a data matrix `X` and a target dimension `k`, project the centered data onto the top `k` principal components.

PCA finds directions where the data varies the most. The projection step is:

```python
X_projected = X_centered @ W
```

where `W` contains the top `k` eigenvectors of the covariance matrix.

---

## Official Problem Statement
Implement a NumPy-based PCA helper that projects input data onto the top `k` principal components.

---

## Constraints
- `X` should be a 2D array-like collection with shape `(n_samples, n_features)`.
- `k` is the number of principal components to keep.
- Convert `X` to a NumPy float array before computation.
- Center the data by subtracting the column-wise mean.
- Compute the sample covariance matrix using `n_samples - 1`.
- Sort eigenvectors by descending eigenvalues.
- Return the projected data as a Python list.

---

## Example

### Input
```python
X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
k = 1
result = pca_projection(X, k)
```

### Output
```python
[[-2.0], [-1.0], [0.0], [1.0], [2.0]]
```

The data only varies along the first feature, so the first principal component captures that direction.

---

## Solution Overview

The implementation in `solution.py`:

- converts `X` to a NumPy float array
- computes the column-wise mean with `np.mean(X, axis=0)`
- centers the data by subtracting the mean
- computes the sample covariance matrix manually
- performs eigen decomposition with `np.linalg.eig`
- sorts eigenvalues in descending order using `np.argsort`
- reorders eigenvectors to match that sorted order
- slices the top `k` eigenvectors into `W`
- projects centered data with `x_c @ W`
- returns the projection as a Python list

---

## Usage

```python
from solution import pca_projection

X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
projected = pca_projection(X, k=1)
print(projected)
```
