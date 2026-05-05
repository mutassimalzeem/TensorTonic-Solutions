# Explanation for TT-024 - PCA Projection

PCA is a dimensionality reduction technique. It rotates the data into directions of maximum variance, then keeps only the strongest directions.

This implementation focuses on the projection part: take `X`, find the top `k` principal directions, and return the lower-dimensional representation.

## Thinking process

The comments in `solution.py` show the path clearly:

1. First, think about the mean feature by feature.
2. Replace manual mean loops with `np.mean(X, axis=0)`.
3. Center the data so PCA measures variation around the origin.
4. Build the covariance matrix from the centered data.
5. Use eigen decomposition to find variance directions.
6. Sort eigenvectors by their eigenvalues.
7. Keep the first `k` eigenvectors.
8. Multiply centered data by those vectors to get the projection.

## Centering

PCA should be applied to centered data:

```python
x_mean = np.mean(X, axis=0)
x_c = X - x_mean
```

Without centering, the result can be pulled toward the raw location of the data instead of the directions where it varies.

## Covariance matrix

The covariance matrix is computed as:

```python
c = (1 / (np.shape(X)[0] - 1)) * np.transpose(x_c) @ x_c
```

For `n_samples` rows, this is the sample covariance formula:

```python
C = (X_centered.T @ X_centered) / (n_samples - 1)
```

## Eigen decomposition

The code uses:

```python
eigenvalues, eigenvectors = np.linalg.eig(c)
```

Eigenvalues tell how much variance each direction explains. Eigenvectors give the directions themselves.

The eigenvalues are sorted from largest to smallest:

```python
idx = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, idx]
```

Then the top `k` eigenvectors are selected:

```python
W = sorted_eigenvectors[:, :k]
```

## Projection

The final projection is:

```python
x_proj = x_c @ W
```

If `X` has shape `(n_samples, n_features)` and `W` has shape `(n_features, k)`, then:

```python
x_proj.shape == (n_samples, k)
```

The function returns this result as a nested Python list.

## Sign note

Eigenvector signs can flip without changing the PCA direction. For some datasets, a valid PCA implementation may return the same projection multiplied by `-1`.
