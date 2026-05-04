# TT-023 - K-Means Clustering

## Problem Link
TensorTonic: K-Means Clustering

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the assignment step of **K-means clustering**. Given a set of points and a set of centroids, assign each point to the index of its nearest centroid.

This implementation compares squared Euclidean distances:

```python
distance = sum((point - centroid) ** 2)
```

The square root is not needed because squared distances preserve the same nearest-centroid ordering.

---

## Official Problem Statement
Implement a NumPy-based K-means helper that returns the nearest centroid label for each input point.

---

## Constraints
- `points` should be a 2D array-like collection of data points.
- `centroids` should be a 2D array-like collection of centroid positions.
- Each point and centroid should have the same number of features.
- Return a flat Python list of integer centroid indices.
- The output length should equal `len(points)`.
- If two centroids have the same distance from a point, `np.argmin` returns the first matching centroid index.

---

## Example

### Input
```python
points = [[0, 0], [5, 5], [1, 1]]
centroids = [[0, 0], [5, 5]]
labels = k_means_assignment(points, centroids)
```

### Output
```python
[0, 1, 0]
```

The first and third points are closer to centroid `0`, while `[5, 5]` is closest to centroid `1`.

---

## Solution Overview

The implementation in `solution.py`:

- defines a `distance` helper for squared Euclidean distance
- loops through every point
- computes the distance from that point to each centroid
- uses `np.argmin` to find the nearest centroid index
- converts the nearest index to a Python `int`
- appends each index to `labels`
- returns the flat list of labels

---

## Usage

```python
from solution import k_means_assignment

points = [[0, 0], [5, 5], [1, 1]]
centroids = [[0, 0], [5, 5]]

labels = k_means_assignment(points, centroids)
print(labels)
```
