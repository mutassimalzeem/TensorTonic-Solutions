# Explanation for TT-022 - KNN Distance

K-nearest neighbors compares each test point with all available training points. This implementation returns the training indices of the closest points, not class labels or predictions.

## Distance formula

For two points `x1` and `y1`, the Euclidean distance is:

```python
sqrt(sum((x1 - y1) ** 2))
```

In `solution.py`, that is implemented as:

```python
np.sqrt(np.sum((np.array(x1) - np.array(y1)) ** 2))
```

## Steps in `knn_distance`

1. Convert `X_train` and `X_test` to NumPy arrays.
2. If `X_test` is empty, return an empty integer array with shape `(0, k)`.
3. For each test point, compute distances to every training point.
4. Store each distance with its training index as `(dist, i)`.
5. Sort the pairs by distance.
6. Take the first `k` indices.
7. If there are fewer than `k` training points, append `-1`.
8. Return all neighbor lists as a NumPy integer array.

## Tie behavior

The code sorts by distance only:

```python
distances.sort(key=lambda x: x[0])
```

Python sorting is stable, so if two training points have the same distance, the lower original training index remains earlier.

## Output shape

For `m = len(X_test)`, the output shape is:

```python
(m, k)
```

Each row contains the nearest training indices for one test point.
