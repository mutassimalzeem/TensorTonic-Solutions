# TT-022 - KNN Distance

## Problem Link
TensorTonic: KNN Distance

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the distance lookup part of **k-nearest neighbors**. For each test point, compute its Euclidean distance from every training point and return the indices of the `k` closest training samples.

The Euclidean distance between two points is:

```python
distance = sqrt(sum((x1 - x2) ** 2))
```

---

## Official Problem Statement
Implement a NumPy-based KNN helper that returns the indices of the nearest training points for each test point using Euclidean distance.

---

## Constraints
- `X_train` should be a 2D array-like collection of training points.
- `X_test` should be a 2D array-like collection of test points.
- `k` is the number of nearest training indices to return for each test point.
- Return an integer NumPy array with shape `(len(X_test), k)`.
- If `X_test` is empty, return an empty integer array with shape `(0, k)`.
- If `k` is larger than the number of training samples, fill missing neighbor positions with `-1`.

---

## Example

### Input
```python
X_train = [[0, 0], [2, 0], [0, 2]]
X_test = [[1, 0]]
k = 2
```

### Output
```python
array([[0, 1]])
```

Training point `0` is distance `1.0` from `[1, 0]`, and training point `1` is also distance `1.0`. With equal distances, the current implementation keeps the earlier training index first.

---

## Solution Overview

The implementation in `solution.py`:

- converts `X_train` and `X_test` to NumPy arrays
- returns `np.empty((0, k), dtype=int)` when there are no test points
- loops through each test point
- computes Euclidean distance to every training point
- stores `(distance, index)` pairs
- sorts by distance
- collects the first `k` indices
- pads with `-1` if fewer than `k` training samples exist
- returns the result as an integer NumPy array

---

## Usage

```python
from solution import knn_distance

X_train = [[0, 0], [2, 0], [0, 2]]
X_test = [[1, 0]]

indices = knn_distance(X_train, X_test, k=2)
print(indices)
```
