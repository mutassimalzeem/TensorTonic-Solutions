# Explanation for TT-023 - K-Means Clustering

K-means clustering groups points by repeatedly assigning each point to its nearest centroid and then updating the centroid positions. This implementation covers the assignment step only.

## Distance formula

For a point `x1` and centroid `y1`, squared Euclidean distance is:

```python
sum((x1 - y1) ** 2)
```

In `solution.py`, that is implemented as:

```python
np.sum((np.array(x1) - np.array(y1))**2)
```

The usual Euclidean distance includes a square root, but the square root is skipped here because it does not change which centroid is nearest.

## Steps in `k_means_assignment`

1. Create an empty `labels` list.
2. Loop through each point.
3. Compute the squared distance from the point to every centroid.
4. Store those distances in `dist`.
5. Use `np.argmin(dist)` to find the nearest centroid.
6. Convert the result to a Python integer.
7. Append the centroid index to `labels`.
8. Return the full label list.

## Tie behavior

`np.argmin` returns the first minimum value. If two centroids are equally close to a point, the lower centroid index is chosen.

## Output shape

For `n = len(points)`, the output is:

```python
len(labels) == n
```

Each value in `labels` is the index of the nearest centroid for the matching input point.
